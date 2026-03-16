import dataclasses

from miyagi.protocols.model import BaseModelArgs


@dataclasses.dataclass
class ModelArgs(BaseModelArgs):
  pretrained_model_name_or_path: str = "black-forest-labs/FLUX.2-klein-9B"
  text_encoder_model_name_or_path: str | None = None
  tokenizer_model_name_or_path: str | None = None
  enable_gradient_checkpointing: bool = False

  lora_rank: int = 16
  lora_alpha: int = 16
  lora_target_modules: list[str] = dataclasses.field(default_factory=lambda: [])
  include_final_layer_lora: bool = False
  strict_lora_target_match: bool = True
  lora_dropout: float = 0.0
  lora_initialisation_style: str = "gaussian"
  offload_vae: bool = False
  offload_text_encoder: bool = False
  num_transformer_blocks_to_swap: int = 0

  timestep_sampling: str = "logit_normal"
  sigmoid_scale: float = 1.0
  logit_mean: float = 0.0
  logit_std: float = 1.0
  discrete_flow_shift: float | None = 3.0
  flow_schedule_auto_shift: bool = False
  guidance_scale: float = 1.0
  max_sequence_length: int = 512
  text_encoder_out_layers: list[int] = dataclasses.field(default_factory=lambda: [9, 18, 27])

  pretrain_loss_weight: float = 1.0
  refl_loss_weight: float = -1.0
  noise_scheduler_timesteps: int = 50
  noise_scheduler_timestep_range_start: int = 0
  noise_scheduler_timestep_range_end: int = 20
