#!/usr/bin/env python3
"""
Standalone LoRA key converter for ComfyUI compatibility.

Converts LoRA safetensors between different naming schemes:
  - Klein format (base_model.model.*) <-> ComfyUI format (diffusion_model.*)
  - FAL.AI format (lora_unet_*) <-> ComfyUI format (norm_out/proj_out)

Usage:
  # Convert a single file (auto-detects format)
  python convert_lora.py input.safetensors

  # Convert a single file to a specific output path
  python convert_lora.py input.safetensors -o output.safetensors

  # Convert all safetensors in a directory
  python convert_lora.py /path/to/lora_folder/

  # Convert a directory, output to a different directory
  python convert_lora.py /path/to/lora_folder/ -o /path/to/output_folder/

  # Dry run (shows what would be converted without writing files)
  python convert_lora.py input.safetensors --dry-run

Requirements:
  pip install torch safetensors
"""

import argparse
import glob
import hashlib
import json
import os
import tempfile
import sys
from datetime import datetime, timezone
from urllib.parse import urlparse
from urllib.request import urlopen


def detect_lora_format(state_dict):
    """Detect the LoRA format based on key naming patterns."""
    fal_forward_keys = {
        "lora_unet_final_layer_adaLN_modulation_1.lora_down.weight",
        "lora_unet_final_layer_adaLN_modulation_1.lora_up.weight",
        "lora_unet_final_layer_linear.lora_down.weight",
        "lora_unet_final_layer_linear.lora_up.weight",
    }
    fal_reverse_keys = {
        "norm_out.linear.lora_A.weight",
        "norm_out.linear.lora_B.weight",
        "proj_out.lora_A.weight",
        "proj_out.lora_B.weight",
    }

    if any(k in state_dict for k in fal_forward_keys):
        return "fal"
    if any(k in state_dict for k in fal_reverse_keys):
        return "fal_reverse"
    if any(k.startswith("base_model.model.") for k in state_dict):
        return "klein"
    if any(k.startswith("diffusion_model.") for k in state_dict):
        return "comfyui"
    return "unknown"


def convert_klein(state_dict, forward=True):
    """Convert between Klein (base_model.model.*) and ComfyUI (diffusion_model.*) formats."""
    if forward:
        prefix_old, prefix_new = "base_model.model.", "diffusion_model."
    else:
        prefix_old, prefix_new = "diffusion_model.", "base_model.model."

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix_old):
            new_key = prefix_new + key[len(prefix_old):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def convert_fal(state_dict, forward=True):
    """Convert between FAL.AI (lora_unet_*) and ComfyUI (norm_out/proj_out) formats."""
    import torch

    key_map = {
        "lora_unet_final_layer_adaLN_modulation_1.lora_down.weight": "norm_out.linear.lora_A.weight",
        "lora_unet_final_layer_adaLN_modulation_1.lora_up.weight": "norm_out.linear.lora_B.weight",
        "lora_unet_final_layer_linear.lora_down.weight": "proj_out.lora_A.weight",
        "lora_unet_final_layer_linear.lora_up.weight": "proj_out.lora_B.weight",
    }
    if not forward:
        key_map = {v: k for k, v in key_map.items()}

    new_state_dict = {}
    for key, value in state_dict.items():
        if key in key_map:
            new_key = key_map[key]
            if forward and "adaLN_modulation" in key:
                shift, scale = value.chunk(2, dim=0)
                value = torch.cat([scale, shift], dim=0)
            elif not forward and "adaLN_modulation" in new_key:
                scale, shift = value.chunk(2, dim=0)
                value = torch.cat([shift, scale], dim=0)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def convert_file(input_path, output_path, dry_run=False):
    """Convert a single safetensors file. Returns (format_detected, success)."""
    from safetensors.torch import load_file, save_file

    state_dict = load_file(input_path)
    fmt = detect_lora_format(state_dict)
    basename = os.path.basename(input_path)

    format_labels = {
        "klein": "Klein (base_model.model.*) -> ComfyUI (diffusion_model.*)",
        "comfyui": "ComfyUI (diffusion_model.*) -> Klein (base_model.model.*)",
        "fal": "FAL.AI (lora_unet_*) -> ComfyUI (norm_out/proj_out)",
        "fal_reverse": "ComfyUI (norm_out/proj_out) -> FAL.AI (lora_unet_*)",
        "unknown": "Unknown format - no conversion applied",
    }
    print(f"  {basename}: {format_labels[fmt]}")

    if fmt == "unknown":
        print(f"  Skipping {basename} - no convertible keys found.")
        return fmt, False

    if dry_run:
        print(f"  [Dry run] Would save to: {output_path}")
        return fmt, True

    if fmt == "klein":
        new_state_dict = convert_klein(state_dict, forward=True)
    elif fmt == "comfyui":
        new_state_dict = convert_klein(state_dict, forward=False)
    elif fmt == "fal":
        new_state_dict = convert_fal(state_dict, forward=True)
    elif fmt == "fal_reverse":
        new_state_dict = convert_fal(state_dict, forward=False)

    save_file(new_state_dict, output_path)
    print(f"  Saved to: {output_path}")
    return fmt, True


def is_url(value):
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and parsed.netloc != ""


def resolve_download_path(url, output_arg):
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path) or "downloaded_lora.safetensors"
    if not filename.endswith(".safetensors"):
        filename = f"{filename}.safetensors"

    if output_arg:
        output_abs = os.path.abspath(output_arg)
        if output_abs.endswith(".safetensors"):
            return output_abs
        os.makedirs(output_abs, exist_ok=True)
        return os.path.join(output_abs, filename)

    return os.path.join(tempfile.gettempdir(), filename)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url, destination, dry_run=False):
    if dry_run:
        print(f"  [Dry run] Would download URL to: {destination}")
        return destination, False

    if os.path.exists(destination):
        print(f"  Reusing existing download: {destination}")
        return destination, False

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    print(f"  Downloading: {url}")
    with urlopen(url, timeout=120) as response, open(destination, "wb") as out_file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)
    print(f"  Downloaded to: {destination}")
    return destination, True


def write_manifest(path, rows):
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "count": len(rows),
        "items": rows,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Manifest written to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LoRA safetensors between naming schemes for ComfyUI compatibility.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
supported formats:
  Klein     base_model.model.*  (e.g. from Klein trainer)
  ComfyUI   diffusion_model.*   (native ComfyUI format)
  FAL.AI    lora_unet_*         (from FAL.AI training)

The format is auto-detected. Conversion direction is chosen automatically:
  Klein   -> ComfyUI
  ComfyUI -> Klein
  FAL.AI  -> ComfyUI (with tensor reordering)
  ComfyUI -> FAL.AI  (with tensor reordering)
""",
    )
    parser.add_argument(
        "input",
        help="Path to a .safetensors file/directory, or an http(s) URL to a .safetensors file.",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (single file) or directory (batch). "
             "Defaults to <name>_converted.safetensors next to the input.",
    )
    parser.add_argument(
        "--suffix",
        default="_converted",
        help="Suffix appended to output filenames in batch mode (default: '_converted').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without writing any files.",
    )
    parser.add_argument(
        "--manifest",
        help="Optional JSON path to write conversion metadata for automation.",
    )
    args = parser.parse_args()

    manifest_rows = []
    downloaded_input = None
    input_value = args.input

    if is_url(input_value):
        download_path = resolve_download_path(input_value, args.output)
        downloaded_input, _ = download_file(input_value, download_path, dry_run=args.dry_run)
        input_path = os.path.abspath(downloaded_input)
    else:
        input_path = os.path.abspath(input_value)
        if not os.path.exists(input_path):
            print(f"Error: {input_path} does not exist.")
            sys.exit(1)

    # Build file list
    if os.path.isfile(input_path):
        if not input_path.endswith(".safetensors"):
            print("Error: Input file must be a .safetensors file.")
            sys.exit(1)
        files = [input_path]
    elif os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.safetensors")))
        if not files:
            print(f"Error: No .safetensors files found in {input_path}")
            sys.exit(1)
    else:
        print(f"Error: {input_path} is not a file or directory.")
        sys.exit(1)

    # Determine output paths
    output_paths = []
    if len(files) == 1 and args.output and not os.path.isdir(args.output):
        # Single file with explicit output path
        output_paths.append(os.path.abspath(args.output))
    else:
        # Batch mode or single file without explicit output
        if args.output:
            out_dir = os.path.abspath(args.output)
        else:
            out_dir = os.path.dirname(files[0]) if os.path.isfile(input_path) else input_path
        if not args.dry_run:
            os.makedirs(out_dir, exist_ok=True)
        for f in files:
            name, ext = os.path.splitext(os.path.basename(f))
            output_paths.append(os.path.join(out_dir, f"{name}{args.suffix}{ext}"))

    # Convert
    print(f"Processing {len(files)} file(s)...")
    converted = 0
    skipped = 0
    errors = 0

    for fpath, opath in zip(files, output_paths):
        item = {
            "input_path": fpath,
            "output_path": opath,
            "input_sha256": None,
            "output_sha256": None,
            "format_detected": None,
            "status": "unknown",
        }
        if os.path.exists(opath) and not args.dry_run:
            print(f"  Skipping {os.path.basename(fpath)} - output already exists.")
            skipped += 1
            item["status"] = "skipped_existing_output"
            item["input_sha256"] = sha256_file(fpath)
            item["output_sha256"] = sha256_file(opath)
            manifest_rows.append(item)
            continue
        try:
            fmt, success = convert_file(fpath, opath, dry_run=args.dry_run)
            item["format_detected"] = fmt
            if success:
                converted += 1
                item["status"] = "converted"
                if not args.dry_run:
                    item["input_sha256"] = sha256_file(fpath)
                    item["output_sha256"] = sha256_file(opath)
            else:
                skipped += 1
                item["status"] = "skipped_unknown_format"
        except Exception as e:
            print(f"  Error converting {os.path.basename(fpath)}: {e}")
            errors += 1
            item["status"] = "error"
            item["error"] = str(e)
        manifest_rows.append(item)

    if args.manifest:
        manifest_path = os.path.abspath(args.manifest)
        manifest_dir = os.path.dirname(manifest_path)
        if manifest_dir:
            os.makedirs(manifest_dir, exist_ok=True)
        write_manifest(manifest_path, manifest_rows)

    if downloaded_input and not args.dry_run:
        print(f"Downloaded input file: {downloaded_input}")

    print(f"\nDone. Converted: {converted}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
