import contextlib
import math
from pprint import pformat
from typing import cast

import torch
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Transformer2DModel
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
from diffusers.pipelines.flux2.pipeline_flux2_klein import retrieve_timesteps
from loguru import logger
from peft import LoraConfig
from tqdm.auto import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from miyagi.components.observability.perf_tracker import trace
from miyagi.config.job_config import JobConfig
from miyagi.distributed.utils import is_rank0, wait_for_everyone
from miyagi.engine.trainer import Runner
from miyagi.utils.other import extract_model_from_parallel, offload_models


from data_sample import DataSample  # isort:skip
from model.model_args_flux_klein import ModelArgs  # isort:skip
from model.parallelize_flux_klein import parallelize_text_encoder  # isort:skip
from model.reward_models.MIMR.multi_id_matching_reward import MIMR  # isort:skip


def mse_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
  return torch.nn.functional.mse_loss(pred.float(), labels.float().detach())


def build_mse_loss(job_config: JobConfig, **kwargs):
  del kwargs
  loss_fn = mse_loss
  if job_config.compile.enable and "loss" in job_config.compile.components:
    logger.info("Compiling the loss function with torch.compile")
    loss_fn = torch.compile(loss_fn, backend=job_config.compile.backend)
  return loss_fn


def time_shift(mu: float, sigma: float, t: torch.Tensor):
  return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
  m = (y2 - y1) / (x2 - x1)
  b = y1 - m * x1
  return lambda x: m * x + b


class UmoFluxKleinRunner(Runner):
  text_encoding_pipeline: Flux2KleinPipeline
  vae: AutoencoderKLFlux2
  noise_scheduler: FlowMatchEulerDiscreteScheduler
  mmir_reward_model: MIMR
  adapter_target_modules: list[str] = ["Flux2TransformerBlock", "Flux2SingleTransformerBlock"]

  def __init__(self, config: JobConfig):
    super().__init__(config)

  def _get_text_encoder_model_path(self, model_args: ModelArgs) -> str:
    return model_args.text_encoder_model_name_or_path or model_args.pretrained_model_name_or_path

  def _get_tokenizer_model_path(self, model_args: ModelArgs) -> str:
    return model_args.tokenizer_model_name_or_path or model_args.pretrained_model_name_or_path

  def _guidance(self, batch_size: int) -> torch.Tensor | None:
    model = extract_model_from_parallel(self.wrapped_model[0])
    if not model.config.guidance_embeds:
      return None
    guidance = torch.full((batch_size,), self.train_spec.model_args.guidance_scale, device=self.device)
    return guidance.to(self.compute_dtype)

  def _normalize_patchified_latents(self, latents: torch.Tensor) -> torch.Tensor:
    latents = Flux2KleinPipeline._patchify_latents(latents)
    return (latents - self.vae.latents_bn_mean) / self.vae.latents_bn_std

  def _decode_patchified_latents(self, latents: torch.Tensor) -> torch.Tensor:
    latents = latents * self.vae.latents_bn_std + self.vae.latents_bn_mean
    latents = Flux2KleinPipeline._unpatchify_latents(latents)
    return latents.to(self.vae.dtype)

  def _sample_timesteps(self, batch_size: int, latent_height: int, latent_width: int) -> torch.Tensor:
    model_args: ModelArgs = self.train_spec.model_args
    if model_args.timestep_sampling == "logit_normal":
      dist = torch.distributions.normal.Normal(0, 1)
    elif model_args.timestep_sampling == "uniform":
      dist = torch.distributions.uniform.Uniform(0, 1)
    else:
      raise NotImplementedError(f"Unsupported timestep sampling {model_args.timestep_sampling!r}")

    t = dist.sample((batch_size,)).to(self.device)
    if model_args.timestep_sampling == "logit_normal":
      t = t * model_args.sigmoid_scale
      t = torch.sigmoid(t)

    if shift := model_args.discrete_flow_shift:
      t = (t * shift) / (1 + (shift - 1) * t)
    elif model_args.flow_schedule_auto_shift:
      mu = get_lin_function(y1=0.5, y2=1.15)((latent_height // 2) * (latent_width // 2))
      t = time_shift(mu, 1.0, t)
    return t

  def _prepare_control_inputs(
    self, model_input_ids: torch.Tensor, control_latents: list[torch.Tensor]
  ) -> tuple[torch.Tensor, torch.Tensor]:
    packed_control_inputs = []
    control_input_ids = []
    for sample_control_latents in control_latents:
      sample_refs = [sample_control_latents[i].unsqueeze(0) for i in range(sample_control_latents.shape[0])]
      packed_refs = [Flux2KleinPipeline._pack_latents(ref).squeeze(0) for ref in sample_refs]
      packed_control_inputs.append(torch.cat(packed_refs, dim=0).unsqueeze(0))
      control_ids = Flux2KleinPipeline._prepare_image_ids(sample_refs).to(device=model_input_ids.device)
      control_ids = control_ids.view(1, -1, model_input_ids.shape[-1])
      control_input_ids.append(control_ids)
    return torch.cat(packed_control_inputs, dim=0), torch.cat(control_input_ids, dim=0)

  def _prepare_transformer_inputs(
    self, packed_image_latents: torch.Tensor, model_input_ids: torch.Tensor, control_latents: list[torch.Tensor]
  ) -> tuple[torch.Tensor, torch.Tensor]:
    packed_control_latents, control_input_ids = self._prepare_control_inputs(model_input_ids, control_latents)
    latent_model_input = torch.cat([packed_image_latents, packed_control_latents], dim=1)
    image_input_ids = torch.cat([model_input_ids, control_input_ids], dim=1)
    return latent_model_input, image_input_ids

  def _resolve_all_linear_target_modules(self, transformer_model: torch.nn.Module) -> list[str]:
    target_linear_modules = set()
    for name, module in transformer_model.named_modules():
      if module.__class__.__name__ not in self.adapter_target_modules:
        continue
      for full_submodule_name, submodule in module.named_modules(prefix=name):
        if isinstance(submodule, torch.nn.Linear):
          target_linear_modules.add(full_submodule_name)
    resolved = sorted(target_linear_modules)
    if not resolved:
      raise ValueError("No linear modules found for all-linear LoRA targeting.")
    return resolved

  def _resolve_explicit_target_modules(self, transformer_model: torch.nn.Module, model_args: ModelArgs) -> list[str]:
    requested_targets = list(model_args.lora_target_modules)
    if model_args.include_final_layer_lora:
      requested_targets.extend(["norm_out.linear", "proj_out"])
    requested_targets = list(dict.fromkeys(requested_targets))
    if not requested_targets:
      raise ValueError("lora_target_modules is empty and all-linear mode is not enabled.")

    available_linear_modules = [
      name for name, module in transformer_model.named_modules() if isinstance(module, torch.nn.Linear)
    ]
    resolved = []
    missing = []
    for target in requested_targets:
      matches = [
        module_name
        for module_name in available_linear_modules
        if module_name == target or module_name.endswith(target) or module_name.endswith(f".{target}")
      ]
      if matches:
        resolved.extend(matches)
      else:
        missing.append(target)
    resolved = sorted(set(resolved))

    if missing:
      message = (
        f"LoRA target modules not found: {missing}. "
        f"Sample available linear modules: {available_linear_modules[:20]}"
      )
      if model_args.strict_lora_target_match:
        raise ValueError(message)
      logger.warning(message)

    if not resolved:
      raise ValueError("No LoRA target modules resolved from requested targets.")

    logger.info(f"Requested LoRA target modules: {requested_targets}")
    logger.info(f"Resolved LoRA target modules ({len(resolved)}): {resolved}")
    return resolved

  def _init_training_components(self):
    model_args: ModelArgs = self.train_spec.model_args

    text_encoder_path = self._get_text_encoder_model_path(model_args)
    tokenizer_path = self._get_tokenizer_model_path(model_args)

    logger.info(
      f"Loading text encoder| path: {text_encoder_path!r}; torch_dtype: {self.torch_dtype!r}; offload: {model_args.offload_text_encoder}"
    )
    text_encoder = Qwen3ForCausalLM.from_pretrained(
      text_encoder_path, subfolder="text_encoder", torch_dtype=self.torch_dtype
    )
    if self.parallel_dims.fsdp_enabled:
      text_encoder = parallelize_text_encoder(
        text_encoder, self.parallel_dims, self.config, offload=model_args.offload_text_encoder
      )
      if not model_args.offload_text_encoder:
        text_encoder = text_encoder.to(self.device)
    else:
      te_device = self.device if not model_args.offload_text_encoder else "cpu"
      text_encoder.to(te_device)
    text_encoder.eval().requires_grad_(False)

    tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path, subfolder="tokenizer")
    self.text_encoding_pipeline = Flux2KleinPipeline.from_pretrained(
      model_args.pretrained_model_name_or_path,
      vae=None,
      transformer=None,
      tokenizer=tokenizer,
      text_encoder=text_encoder,
      scheduler=None,
      torch_dtype=self.torch_dtype,
    )

    logger.info(
      f"Loading vae| path: {model_args.pretrained_model_name_or_path!r}; torch_dtype: {self.torch_dtype!r}; offload: {model_args.offload_vae}"
    )
    vae = AutoencoderKLFlux2.from_pretrained(
      model_args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=self.torch_dtype
    )
    vae.eval().requires_grad_(False)
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(dtype=vae.dtype)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(dtype=vae.dtype)
    vae.register_buffer("latents_bn_mean", latents_bn_mean)
    vae.register_buffer("latents_bn_std", latents_bn_std)

    if model_args.offload_vae:
      raise NotImplementedError("Offloading VAE is not working as expected.")

    vae_device = self.device if not model_args.offload_vae else "cpu"
    vae.to(vae_device)
    self.vae = vae

    self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
      model_args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    mmir_reward_model = MIMR(self.device, self.torch_dtype)
    mmir_reward_model.eval().requires_grad_(False)
    self.mmir_reward_model = mmir_reward_model
    self.mse_loss = build_mse_loss(self.config)

  def prepare_model(self, model: Flux2Transformer2DModel) -> list[torch.nn.Module]:
    if self.parallel_dims.pp_enabled:
      raise NotImplementedError("Pipeline Parallelism is not supported yet")

    if self.train_spec.model_args.num_transformer_blocks_to_swap > 0:
      raise NotImplementedError("Transformer block swapping is not supported for Flux Klein yet")

    if not self.parallel_dims.fsdp_enabled:
      model_module = cast(torch.nn.Module, model)
      logger.info(f"casting model to {self.torch_dtype}")
      model_module.to(self.torch_dtype)

      if self.parallel_dims.dp_replicate_enabled:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model_module.to(self.device)
        if self.config.compile.enable and "model" in self.config.compile.components:
          torch._dynamo.config.optimize_ddp = "ddp_optimizer"

        with torch.cuda.stream(torch.cuda.Stream()):
          model_module = cast(
            torch.nn.Module,
            DDP(model_module, device_ids=[self.device.index], output_device=self.device, bucket_cap_mb=100),
          )
        logger.info("Applied DDP to the model")

      wait_for_everyone()
      return [model_module]

    model.to(self.device)
    wrapped_model = self.train_spec.parallelize_fn(model, self.parallel_dims, self.config)
    del model
    wrapped_model.train()
    wait_for_everyone()
    return [wrapped_model]

  def get_model(self) -> torch.nn.Module:
    model_args: ModelArgs = self.train_spec.model_args
    logger.info(f"Building model with args: {model_args}; torch_dtype: {self.torch_dtype!r}")

    transformer_model = Flux2Transformer2DModel.from_pretrained(
      model_args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=self.torch_dtype
    )
    transformer_model.requires_grad_(False)

    if model_args.lora_target_modules and model_args.lora_target_modules[0] == "all-linear":
      target_linear_modules = self._resolve_all_linear_target_modules(transformer_model)
      logger.info(f"Adding LoRA to the following linear modules: \n\t{pformat(target_linear_modules, compact=True)}")
      transformer_lora_config = LoraConfig(
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        init_lora_weights=model_args.lora_initialisation_style,
        target_modules=target_linear_modules,
        lora_dropout=model_args.lora_dropout,
      )
    else:
      resolved_target_modules = self._resolve_explicit_target_modules(transformer_model, model_args)
      transformer_lora_config = LoraConfig(
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        init_lora_weights=model_args.lora_initialisation_style,
        target_modules=resolved_target_modules,
        lora_dropout=model_args.lora_dropout,
      )
    transformer_model.add_adapter(transformer_lora_config)

    for name, param in transformer_model.named_parameters():
      if "lora" in name.lower():
        param.requires_grad_(True)
      else:
        param.requires_grad_(False)

    if model_args.enable_gradient_checkpointing:
      transformer_model.enable_gradient_checkpointing()
      logger.info("[ Diffusers ] gradient checkpointing enabled")

    if "model" in self.config.compile.components and self.config.compile.enable:
      transformer_model.compile_repeated_blocks()
      logger.info("[ Diffusers ] model compile_repeated_blocks enabled")

    logger.info(f"Model:\n{transformer_model}")

    lora_param_count = sum(
      p.numel() for n, p in transformer_model.named_parameters() if "lora" in n.lower() and p.requires_grad
    )
    trainable_params = sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)
    param_count = sum(p.numel() for p in transformer_model.parameters())

    logger.info(f"Added LoRA adapter with rank {model_args.lora_rank}")
    logger.info(f"Number of parameters: {param_count / 1e6:.2f} million")
    logger.info(f"Target modules: {model_args.lora_target_modules}")
    logger.info(
      f"LoRA parameters: {lora_param_count / 1e6:.2f}M ({100 * lora_param_count / param_count:.2f}% of total)"
    )
    logger.info(f"Total trainable parameters: {trainable_params / 1e6:.2f}M")

    self._init_training_components()
    return transformer_model

  @torch.inference_mode()
  @trace("trainer/train_step/prepare_training_batch")
  def __prepare_training_batch(self, batch: list[DataSample]) -> list[DataSample]:
    model_args: ModelArgs = self.train_spec.model_args

    def _prepare_one_item(data_sample: DataSample) -> DataSample:
      if self.train_spec.model_args.offload_text_encoder:
        ctx = offload_models(
          self.text_encoding_pipeline, device=self.device, offload=self.train_spec.model_args.offload_text_encoder
        )
      else:
        ctx = contextlib.nullcontext()

      with ctx:
        prompt_embeds, text_ids = self.text_encoding_pipeline.encode_prompt(
          prompt=data_sample.prompt,
          device=self.device,
          num_images_per_prompt=1,
          max_sequence_length=model_args.max_sequence_length,
          text_encoder_out_layers=tuple(model_args.text_encoder_out_layers),
        )
      data_sample.prompt_embeds = prompt_embeds.squeeze(0)
      data_sample.text_ids = text_ids.squeeze(0)

      self.vae.eval().requires_grad_(False)
      with offload_models(self.vae, device=self.device, offload=self.train_spec.model_args.offload_vae):
        image_pixel_values = data_sample.pixel_values.unsqueeze(0).to(self.device, self.vae.dtype)
        image_latents = self.vae.encode(image_pixel_values).latent_dist.mode()
        data_sample.image_latents = self._normalize_patchified_latents(image_latents).squeeze(0)

        control_latents_list = []
        if data_sample.init_image.condition_images.shape[0] == 0:
          raise ValueError("Flux Klein training requires at least one reference image per sample.")
        for control_tensor in data_sample.init_image.condition_images:
          control_tensor = (2.0 * control_tensor - 1.0).unsqueeze(0).to(self.device, self.vae.dtype)
          encoded = self.vae.encode(control_tensor).latent_dist.mode()
          control_latent = self._normalize_patchified_latents(encoded)
          control_latents_list.append(control_latent)
        data_sample.control_latents = torch.cat(control_latents_list, dim=0)
      return data_sample

    return [_prepare_one_item(data_sample) for data_sample in batch]

  @trace("trainer/train_step/diffusion_loss")
  def compute_diffusion_loss(self, batch: list[DataSample]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = len(batch)
    transformer = self.wrapped_model[0]
    transformer.train()

    image_latents = torch.stack([sample.image_latents for sample in batch]).to(self.device, self.compute_dtype)
    _, num_channels, latent_height, latent_width = image_latents.shape

    control_latents = [sample.control_latents.to(self.device, self.compute_dtype) for sample in batch]
    prompt_embeds = torch.stack([sample.prompt_embeds for sample in batch]).to(self.device, self.compute_dtype)
    text_ids = torch.stack([sample.text_ids for sample in batch]).to(self.device)

    noise = torch.randn_like(image_latents)
    t = self._sample_timesteps(batch_size, latent_height, latent_width)
    noisy_latents = (1 - t.view(-1, 1, 1, 1)) * image_latents + t.view(-1, 1, 1, 1) * noise

    packed_noisy_latents = Flux2KleinPipeline._pack_latents(noisy_latents)
    model_input_ids = Flux2KleinPipeline._prepare_latent_ids(image_latents).to(self.device)
    latent_model_input, image_input_ids = self._prepare_transformer_inputs(
      packed_noisy_latents, model_input_ids, control_latents
    )

    guidance = self._guidance(batch_size)
    model_pred = transformer(
      hidden_states=latent_model_input,
      timestep=t.expand(latent_model_input.shape[0]).to(self.compute_dtype),
      guidance=guidance,
      encoder_hidden_states=prompt_embeds,
      txt_ids=text_ids,
      img_ids=image_input_ids,
      return_dict=False,
    )[0]
    model_pred = model_pred[:, : packed_noisy_latents.size(1), :]
    model_pred = cast(torch.Tensor, Flux2KleinPipeline._unpack_latents_with_ids(model_pred, model_input_ids))

    target = noise - image_latents
    loss = self.mse_loss(model_pred.float(), target.float())
    loss *= self.train_spec.model_args.pretrain_loss_weight
    del model_pred

    log_vars = {
      "diffusion_loss": loss,
      "image_seq_len": torch.tensor(packed_noisy_latents.size(1), device=self.device),
      "txt_seq_len": torch.tensor(prompt_embeds.size(1), device=self.device),
    }
    return loss, log_vars

  @trace("trainer/train_step/refl_loss")
  def compute_refl_loss(self, batch: list[DataSample]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = len(batch)
    transformer = self.wrapped_model[0]
    transformer.eval()

    image_latents = torch.stack([sample.image_latents for sample in batch]).to(self.device, self.compute_dtype)
    control_latents = [sample.control_latents.to(self.device, self.compute_dtype) for sample in batch]
    prompt_embeds = torch.stack([sample.prompt_embeds for sample in batch]).to(self.device, self.compute_dtype)
    text_ids = torch.stack([sample.text_ids for sample in batch]).to(self.device)

    packed_image_latents = Flux2KleinPipeline._pack_latents(image_latents)
    model_input_ids = Flux2KleinPipeline._prepare_latent_ids(image_latents).to(self.device)

    model_args: ModelArgs = self.train_spec.model_args
    refl_num_inference_steps = model_args.noise_scheduler_timesteps
    refl_sigmas = torch.linspace(1.0, 1 / refl_num_inference_steps, refl_num_inference_steps)

    mu = calculate_shift(
      packed_image_latents.shape[1],
      self.noise_scheduler.config.base_image_seq_len,
      self.noise_scheduler.config.max_image_seq_len,
      self.noise_scheduler.config.base_shift,
      self.noise_scheduler.config.max_shift,
    )
    refl_timesteps, refl_num_inference_steps = retrieve_timesteps(
      self.noise_scheduler,
      num_inference_steps=refl_num_inference_steps,
      device=self.device,
      sigmas=cast(list[float], refl_sigmas.tolist()),
      mu=mu,
    )
    self.noise_scheduler.set_begin_index(0)

    reward_start = model_args.noise_scheduler_timestep_range_start
    reward_end = min(model_args.noise_scheduler_timestep_range_end, len(refl_timesteps) - 2)
    reward_step_idx = torch.randint(reward_start, reward_end + 1, (1,), device=self.device).long().item()
    refl_timesteps_inference = refl_timesteps[: reward_step_idx + 1]
    refl_timestep_final = refl_timesteps[reward_step_idx + 1]
    latents = torch.randn_like(packed_image_latents)
    guidance = self._guidance(batch_size)

    torch.set_grad_enabled(False)
    for t in tqdm(refl_timesteps_inference, dynamic_ncols=True, disable=not is_rank0()):
      latent_model_input, image_input_ids = self._prepare_transformer_inputs(latents, model_input_ids, control_latents)
      noise_pred = transformer(
        hidden_states=latent_model_input,
        timestep=(t / 1000.0).expand(batch_size).to(self.compute_dtype),
        guidance=guidance,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=image_input_ids,
        return_dict=False,
      )[0]
      noise_pred = noise_pred[:, : latents.size(1)]
      # compute the previous noisy sample x_t -> x_t-1
      latents = cast(
        torch.Tensor,
        self.noise_scheduler.step(noise_pred, t, cast(torch.FloatTensor, latents), return_dict=False)[0],
      )
      del noise_pred

    torch.set_grad_enabled(True)
    transformer.train()

    latent_model_input, image_input_ids = self._prepare_transformer_inputs(latents, model_input_ids, control_latents)
    noise_pred = transformer(
      hidden_states=latent_model_input,
      timestep=(refl_timestep_final / 1000.0).expand(batch_size).to(self.compute_dtype),
      guidance=guidance,
      encoder_hidden_states=prompt_embeds,
      txt_ids=text_ids,
      img_ids=image_input_ids,
      return_dict=False,
    )[0]
    noise_pred = noise_pred[:, : latents.size(1)]
    latents = latents - noise_pred * refl_sigmas[reward_step_idx + 1]

    latents = cast(torch.Tensor, Flux2KleinPipeline._unpack_latents_with_ids(latents, model_input_ids))
    latents = self._decode_patchified_latents(latents)
    pred_images = self.vae.decode(latents, return_dict=False)[0].clamp(-1, 1)

    init_images = torch.stack([sample.pixel_values for sample in batch], dim=0)
    rewards = []
    for i in range(batch_size):
      reward = self.mmir_reward_model.score_grad(
        image=pred_images[i].to(self.device, self.compute_dtype),
        image_gt=init_images[i].to(self.device, self.compute_dtype).detach(),
      )
      rewards.append(reward)
    loss = torch.stack(rewards).mean()
    del pred_images

    loss *= self.train_spec.model_args.refl_loss_weight
    log_vars = {
      "refl_loss": loss,
      "refl_start_timestep": refl_timesteps_inference[0].item(),
      "refl_end_timestep": refl_timesteps_inference[-1].item(),
      "refl_reward_timestep": refl_timestep_final.item(),
      "refl_reward_step_idx": torch.tensor(reward_step_idx, device=self.device),
    }
    return loss, log_vars

  @trace("trainer/train_step")
  def train_step(self, batch: list[DataSample]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch = self.__prepare_training_batch(batch)
    loss_refl, log_vars_refl = self.compute_refl_loss(batch)
    loss_diffusion, log_vars_diffusion = self.compute_diffusion_loss(batch)
    loss = loss_diffusion + loss_refl
    log_vars = {**log_vars_diffusion, **log_vars_refl}
    return loss, log_vars
