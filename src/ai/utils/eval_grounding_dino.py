#!/usr/bin/env python3
"""
Evaluate Grounding DINO on the pepper-leaf validation split using COCO metrics.

This script:
1. Loads the YOLO auto-split dataset under detection_yolo/.splits/src_split_v20_s42.
2. Runs Grounding DINO with the caption "a green leaf" on every validation image.
3. Saves predictions in COCO JSON format.
4. Computes mAP with pycocotools against the converted COCO ground truth.

Adjust constants below if your paths differ.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_convert

import groundingdino
from groundingdino.util.inference import load_image, load_model, predict


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATASET_ROOT = Path("detection_yolo/.splits/src_split_v20_s42").resolve()
GROUND_TRUTH_JSON = DATASET_ROOT / "annotations/src_split_v20_s42_val.json"
PREDICTIONS_JSON = DATASET_ROOT / "annotations/grounding_dino_val_predictions.json"
METRICS_CSV = DATASET_ROOT / "annotations/grounding_dino_val_metrics.csv"

# Use the Swin-T config that ships with the pip package.
CONFIG_PATH = (
    Path(groundingdino.__file__).resolve().parent / "config" / "GroundingDINO_SwinT_OGC.py"
)
# Update this if your checkpoint lives elsewhere.
CHECKPOINT_PATH = Path("/home/sydney0716/Desktop/AI/models/groundingdino_swint_ogc.pth")

PROMPT = "a green leaf"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_image_id_lookup(gt_json: Path) -> Dict[str, int]:
    data = json.loads(gt_json.read_text())
    return {entry["file_name"]: entry["id"] for entry in data["images"]}


def generate_predictions() -> List[dict]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model = load_model(str(CONFIG_PATH), str(CHECKPOINT_PATH), device=DEVICE)
    model = model.to(DEVICE)
    model.eval()

    gt_lookup = load_image_id_lookup(GROUND_TRUTH_JSON)
    predictions: List[dict] = []

    image_dir = DATASET_ROOT / "images" / "val"
    label_missing: List[str] = []

    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue

        rel_path = str(image_path.relative_to(DATASET_ROOT))
        image_id = gt_lookup.get(rel_path)
        if image_id is None:
            label_missing.append(rel_path)
            continue

        image_np, image_transformed = load_image(str(image_path))
        boxes, scores, _phrases = predict(
            model=model,
            image=image_transformed,
            caption=PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE,
        )

        if boxes.numel() == 0:
            continue

        height, width = image_np.shape[:2]
        scale = torch.tensor([width, height, width, height], dtype=boxes.dtype)
        boxes_scaled = boxes * scale
        boxes_xywh = box_convert(boxes_scaled, in_fmt="cxcywh", out_fmt="xywh")

        for xywh, score in zip(boxes_xywh, scores):
            predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": 1,
                    "bbox": [float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])],
                    "score": float(score),
                }
            )

    if label_missing:
        print(f"[WARN] Skipped {len(label_missing)} image(s) missing in ground truth JSON: {label_missing[:5]}")

    PREDICTIONS_JSON.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_JSON.write_text(json.dumps(predictions, indent=2))
    print(f"[OK] Saved predictions to {PREDICTIONS_JSON}")
    return predictions


def evaluate_predictions() -> Dict[str, float]:
    coco_gt = COCO(str(GROUND_TRUTH_JSON))
    coco_dt = coco_gt.loadRes(str(PREDICTIONS_JSON))
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats  # Array of 12 metrics
    metrics = {
        "model": "GroundingDINO_SwinT",
        "AP": stats[0],
        "AP50": stats[1],
        "AP75": stats[2],
        "AP_small": stats[3],
        "AP_medium": stats[4],
        "AP_large": stats[5],
        "AR1": stats[6],
        "AR10": stats[7],
        "AR100": stats[8],
        "AR_small": stats[9],
        "AR_medium": stats[10],
        "AR_large": stats[11],
    }
    df = pd.DataFrame([metrics])
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(METRICS_CSV, index=False)
    print(f"[OK] Wrote metrics to {METRICS_CSV}")
    return metrics


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if DEVICE == "cpu":
        print("[INFO] CUDA not available. Running Grounding DINO on CPU (slower).")
    generate_predictions()
    evaluate_predictions()


if __name__ == "__main__":
    main()
