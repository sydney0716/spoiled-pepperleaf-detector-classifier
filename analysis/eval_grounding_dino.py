from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_convert

import groundingdino
from groundingdino.util.inference import load_image, load_model, predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "detection_yolo/.splits/detection_processed_split"

IMAGES_VAL_DIR = DATASET_ROOT / "images/val"
LABELS_VAL_DIR = DATASET_ROOT / "labels/val"

OUTPUT_DIR = PROJECT_ROOT / "runs/detection/grounding_dino"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUND_TRUTH_JSON = OUTPUT_DIR / "val_ground_truth.json"
PREDICTIONS_JSON = OUTPUT_DIR / "val_predictions.json"
METRICS_CSV = OUTPUT_DIR / "val_metrics.csv"

# Model configuration
CONFIG_PATH = PROJECT_ROOT / "models/pretrained/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = PROJECT_ROOT / "models/pretrained/groundingdino_swint_ogc.pth"

PROMPT = "a green leaf"
BOX_THRESHOLD = 0.15
TEXT_THRESHOLD = 0.15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_yolo_label(label_path: Path, img_width: int, img_height: int) -> List[List[float]]:
    if not label_path.exists():
        return []

    boxes = []
    with label_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            # We assume class_id is 0 (or ignored since we only have one class for now)
            # _, x_c, y_c, w_n, h_n = parts
            x_c = float(parts[1])
            y_c = float(parts[2])
            w_n = float(parts[3])
            h_n = float(parts[4])

            # Convert normalized center (x_c, y_c, w_n, h_n) to absolute (x_min, y_min, w, h)
            w = w_n * img_width
            h = h_n * img_height
            x_min = (x_c * img_width) - (w / 2)
            y_min = (y_c * img_height) - (h / 2)

            boxes.append([x_min, y_min, w, h])
    return boxes


def create_coco_ground_truth_from_yolo() -> None:
    print(f"[INFO] Generating COCO ground truth JSON from {LABELS_VAL_DIR}...")
    
    images = []
    annotations = []
    categories = [{"id": 1, "name": "leaf"}]
    
    ann_id_counter = 1
    img_id_counter = 1
    
    # Collect all valid image files
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = sorted([p for p in IMAGES_VAL_DIR.iterdir() if p.suffix.lower() in valid_exts])

    if not image_files:
        raise FileNotFoundError(f"No images found in {IMAGES_VAL_DIR}")

    for img_path in image_files:
        # Load image to get dimensions
        # We use load_image from groundingdino utils which returns (image_source, image)
        # image_source is a numpy array (H, W, 3)
        image_source, _ = load_image(str(img_path))
        h, w, _ = image_source.shape
        
        image_info = {
            "id": img_id_counter,
            "file_name": img_path.name,
            "width": w,
            "height": h
        }
        images.append(image_info)

        # Look for corresponding label file
        label_path = LABELS_VAL_DIR / f"{img_path.stem}.txt"
        boxes = parse_yolo_label(label_path, w, h)
        
        for box in boxes:
            annotations.append({
                "id": ann_id_counter,
                "image_id": img_id_counter,
                "category_id": 1,
                "bbox": box,
                "area": box[2] * box[3],
                "iscrowd": 0
            })
            ann_id_counter += 1
            
        img_id_counter += 1

    coco_output = {
        "info": {
            "description": "COCO-style dataset generated from YOLO labels",
            "version": "1.0",
            "year": 2025,
            "contributor": "",
            "date_created": "2025-11-30"
        },
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    GROUND_TRUTH_JSON.write_text(json.dumps(coco_output, indent=2))
    print(f"[OK] Saved temporary ground truth to {GROUND_TRUTH_JSON}")


def load_image_id_lookup(gt_json: Path) -> Dict[str, int]:
    data = json.loads(gt_json.read_text())
    return {entry["file_name"]: entry["id"] for entry in data["images"]}


def generate_predictions() -> List[dict]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    print(f"[INFO] Loading model from {CHECKPOINT_PATH}...")
    model = load_model(str(CONFIG_PATH), str(CHECKPOINT_PATH), device=DEVICE)
    model = model.to(DEVICE)
    model.eval()

    gt_lookup = load_image_id_lookup(GROUND_TRUTH_JSON)
    predictions: List[dict] = []

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = sorted([p for p in IMAGES_VAL_DIR.iterdir() if p.suffix.lower() in valid_exts])

    print(f"[INFO] Running inference on {len(image_files)} images...")

    for image_path in image_files:
        image_id = gt_lookup.get(image_path.name)
        if image_id is None:
            continue

        image_np, image_transformed = load_image(str(image_path))
        
        # Predict
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
        # Convert from cxcywh to xywh (COCO format)
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

    PREDICTIONS_JSON.write_text(json.dumps(predictions, indent=2))
    print(f"[OK] Saved predictions to {PREDICTIONS_JSON}")
    return predictions


def evaluate_predictions() -> Dict[str, float]:
    print("[INFO] Evaluating predictions...")
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
    df.to_csv(METRICS_CSV, index=False)
    print(f"[OK] Wrote metrics to {METRICS_CSV}")
    return metrics

def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if DEVICE == "cpu":
        print("[INFO] CUDA not available. Running Grounding DINO on CPU (slower).")
    
    # 1. Convert YOLO labels to COCO JSON for evaluation ground truth
    create_coco_ground_truth_from_yolo()
    
    # 2. Run inference
    generate_predictions()
    
    # 3. Evaluate
    evaluate_predictions()


if __name__ == "__main__":
    main()