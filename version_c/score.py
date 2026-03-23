"""
version_c score — character likeness scorer using InsightFace

For each row, compares faces in the output image against the character
reference images (Char 1/2/3) using ArcFace embeddings.

Score per row = mean cosine similarity between output faces and reference faces.
Final score = mean across all rows.

Usage:
    python version_c/score.py \
        --outputs version_c/outputs_v2 \
        --test    version_c/test-multi.json \
        --rows    3

Requirements:
    pip install insightface onnxruntime-gpu numpy opencv-python
"""

import argparse
import json
import os
import sys
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image

try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    print("[ERROR] insightface not installed. Run: pip install insightface onnxruntime-gpu")
    sys.exit(1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_image_np(src: str) -> np.ndarray:
    """Load image from URL or local path → BGR numpy array (for insightface)."""
    if src.startswith("http://") or src.startswith("https://"):
        resp = requests.get(src, timeout=60)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        img = Image.open(src).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def get_embedding(app: FaceAnalysis, img_bgr: np.ndarray) -> np.ndarray | None:
    """Detect largest face and return its ArcFace embedding, or None."""
    faces = app.get(img_bgr)
    if not faces:
        return None
    # pick largest face by bounding box area
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.normed_embedding


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="version_c/outputs_v2",
                        help="Folder containing row_001.png etc.")
    parser.add_argument("--test",    default="version_c/test-multi.json")
    parser.add_argument("--rows",    type=int, default=None,
                        help="Limit to first N rows (default: all)")
    args = parser.parse_args()

    # ------------------------------------------------------------------ init
    print("Initialising InsightFace (buffalo_l)...")
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Ready.\n")

    # ------------------------------------------------------------------ data
    with open(args.test) as f:
        sheet = json.load(f)

    rows = sheet["data"]
    if args.rows:
        rows = rows[: args.rows]

    results = []

    for idx, row in enumerate(rows):
        out_path = os.path.join(args.outputs, f"row_{idx+1:03d}.png")
        if not os.path.exists(out_path):
            print(f"[{idx+1}] MISSING output: {out_path} — skipping")
            continue

        print(f"[{idx+1}/{len(rows)}] scoring {out_path}")

        # load output
        out_img = load_image_np(out_path)
        out_emb = get_embedding(app, out_img)
        if out_emb is None:
            print(f"  No face detected in output — score: 0.0")
            results.append({"row": idx + 1, "score": 0.0, "note": "no face in output"})
            continue

        # load char references
        char_scores = []
        for key in ["Char 1", "Char 2", "Char 3"]:
            url = row.get(key, "").strip()
            if not url:
                continue
            try:
                ref_img = load_image_np(url)
                ref_emb = get_embedding(app, ref_img)
                if ref_emb is None:
                    print(f"  {key}: no face detected in reference")
                    continue
                sim = cosine_sim(out_emb, ref_emb)
                print(f"  {key}: similarity = {sim:.4f}")
                char_scores.append(sim)
            except Exception as e:
                print(f"  {key}: error — {e}")

        if not char_scores:
            row_score = 0.0
            note = "no reference faces detected"
        else:
            row_score = float(np.mean(char_scores))
            note = f"{len(char_scores)} chars scored"

        print(f"  row score: {row_score:.4f}  ({note})")
        results.append({"row": idx + 1, "score": row_score, "note": note})

    # ------------------------------------------------------------------ summary
    scored = [r for r in results if r["score"] > 0]
    mean_score = float(np.mean([r["score"] for r in scored])) if scored else 0.0

    print(f"\n{'='*50}")
    print(f"Rows scored:   {len(scored)} / {len(rows)}")
    print(f"Mean likeness: {mean_score:.4f}")
    print(f"{'='*50}\n")

    # save results
    out_json = os.path.join(args.outputs, "scores.json")
    with open(out_json, "w") as f:
        json.dump({"mean_score": mean_score, "rows": results}, f, indent=2)
    print(f"Saved scores → {out_json}")


if __name__ == "__main__":
    main()
