"""
version_c eval — FAL LoRA character swap evaluation
Reads test-multi.json, runs inference with the converted FAL LoRA,
saves outputs to version_c/outputs/.

Usage:
    python version_c/eval.py \
        --lora version_c/pytorch_lora_weights_converted.safetensors \
        --test  version_c/test-multi.json \
        --out   version_c/outputs \
        --model black-forest-labs/FLUX.2-klein-base-9B
"""

import argparse
import json
import os
from io import BytesIO

import requests
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def resize_to(img: Image.Image, size: int = 1024) -> Image.Image:
    """Resize so the longer edge == size, keep aspect ratio."""
    w, h = img.size
    scale = size / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora",  default="version_c/pytorch_lora_weights_converted.safetensors")
    parser.add_argument("--test",  default="version_c/test-multi.json")
    parser.add_argument("--out",   default="version_c/outputs")
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-9B")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--size",  type=int, default=1024)
    parser.add_argument("--rows",  type=int, default=None,
                        help="Limit to first N rows (default: all 40)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------------------------ model
    print(f"Loading pipeline: {args.model}")
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading LoRA: {args.lora}")
    pipe.load_lora_weights(args.lora)

    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()   # keeps VRAM sane on H100
    print("Pipeline ready.\n")

    # ------------------------------------------------------------------ data
    with open(args.test) as f:
        sheet = json.load(f)

    rows = sheet["data"]
    if args.rows:
        rows = rows[: args.rows]

    print(f"Running {len(rows)} rows → {args.out}\n")

    for idx, row in enumerate(rows):
        print(f"[{idx+1}/{len(rows)}] processing...")

        # --- input image
        input_img = resize_to(load_image_from_url(row["Input Image"]), args.size)

        # --- character reference images (Char 1 required, 2+3 optional)
        char_imgs = []
        for key in ["Char 1", "Char 2", "Char 3"]:
            url = row.get(key, "").strip()
            if url:
                char_imgs.append(resize_to(load_image_from_url(url), args.size))

        prompt = row["Prompt Used"]
        num_chars = row.get("Num Chars", len(char_imgs))

        print(f"  chars: {num_chars}  |  prompt: {prompt[:80]}...")

        # --- inference
        result = pipe(
            prompt=prompt,
            image=input_img,
            control_images=char_imgs,   # character reference images
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.size,
            width=args.size,
        ).images[0]

        # --- save
        out_path = os.path.join(args.out, f"row_{idx+1:03d}.png")
        result.save(out_path)
        print(f"  saved → {out_path}")

    print(f"\nDone. {len(rows)} images saved to {args.out}/")


if __name__ == "__main__":
    main()
