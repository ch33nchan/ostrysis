"""
version_c infer — single-shot FAL LoRA character swap
For use by the Slack agent. Takes any input image + character refs + prompt.

Usage:
    python version_c/infer.py \
        --input  <scene_image_path_or_url> \
        --chars  <char1_path_or_url> [<char2> <char3>] \
        --prompt "In a recreation of image 1, replace X with Y" \
        --out    output.png

All --input and --chars args accept local file paths OR http(s) URLs.
"""

import argparse
import os
import sys
from io import BytesIO

import requests
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------

DEFAULT_LORA  = os.path.join(os.path.dirname(__file__), "pytorch_lora_weights_converted.safetensors")
DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-base-9B"
DEFAULT_OUT   = os.path.join(os.path.dirname(__file__), "outputs_agent", "out.png")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_image(src: str) -> Image.Image:
    """Load an image from a local path or http(s) URL."""
    if src.startswith("http://") or src.startswith("https://"):
        resp = requests.get(src, timeout=60)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        if not os.path.exists(src):
            print(f"[ERROR] File not found: {src}", file=sys.stderr)
            sys.exit(1)
        return Image.open(src).convert("RGB")


def resize_portrait(img: Image.Image, width: int, height: int) -> Image.Image:
    return img.resize((width, height), Image.LANCZOS)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Single-shot FAL LoRA character swap inference.")
    parser.add_argument("--input",    required=True,
                        help="Scene/input image — local path or URL")
    parser.add_argument("--chars",    nargs="+", required=True,
                        help="Character reference image(s) — local path or URL (1-3)")
    parser.add_argument("--prompt",   required=True,
                        help="Swap prompt, e.g. 'Replace the man in blue with ...'")
    parser.add_argument("--out",      default=DEFAULT_OUT,
                        help="Output image path (default: version_c/outputs_agent/out.png)")
    parser.add_argument("--lora",     default=DEFAULT_LORA,
                        help="Path to converted LoRA safetensors")
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument("--steps",    type=int,   default=50)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--width",    type=int,   default=768)
    parser.add_argument("--height",   type=int,   default=1344)
    args = parser.parse_args()

    if len(args.chars) > 3:
        print("[ERROR] Maximum 3 character reference images supported.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------ load
    print(f"Loading pipeline: {args.model}")
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    )
    print(f"Loading LoRA: {args.lora}")
    pipe.load_lora_weights(args.lora)
    pipe.enable_model_cpu_offload()
    print("Pipeline ready.\n")

    # ------------------------------------------------------------------ images
    print(f"Loading input image: {args.input}")
    scene = resize_portrait(load_image(args.input), args.width, args.height)

    char_imgs = []
    for i, src in enumerate(args.chars):
        print(f"Loading char {i+1}: {src}")
        char_imgs.append(resize_portrait(load_image(src), args.width, args.height))

    all_images = [scene] + char_imgs

    # ------------------------------------------------------------------ infer
    print(f"\nPrompt: {args.prompt}")
    print(f"Running inference ({args.steps} steps, guidance {args.guidance})...")

    result = pipe(
        prompt=args.prompt,
        image=all_images,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
    ).images[0]

    # ------------------------------------------------------------------ save
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    result.save(args.out)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
