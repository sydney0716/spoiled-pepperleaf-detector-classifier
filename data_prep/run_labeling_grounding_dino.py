"""
This script uses Grounding DINO to generate initial annotations (pre-labeling) for raw data.
These generated labels serve as a starting point and are intended to be refined through manual verification.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
import torch
from groundingdino.util.inference import load_image, load_model, predict

VALID_EXTS = {".jpg",}

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUTS_DIR = PROJECT_ROOT / "runs/groundingDINO"

DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data/raw_data"

DEFAULT_ANNOTATIONS_DIR = OUTPUTS_DIR
DEFAULT_VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"

MODEL_CONFIG_PATH = PROJECT_ROOT / "models/pretrained/GroundingDINO_SwinT_OGC.py"
MODEL_CHECKPOINT_PATH = PROJECT_ROOT / "models/pretrained/groundingdino_swint_ogc.pth"

PREFERRED_DEVICE = "cuda"
BOX_THRESHOLD = 0.15
TEXT_THRESHOLD = 0.15
PROMPT_CAPTION = "a green leaf"
KEYWORDS = ("leaf",)
CATEGORY_ID = 1
ANNOTATIONS_FILENAME = "grounding_dino_annotations.json"

def collect_images(image_dir: Path) -> List[Path]:
    """Return a sorted list of GroundingDINO-compatible images."""
    candidates = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    return sorted(candidates)

def resolve_device(preferred: str) -> str:
    if preferred == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        return "cpu"
    return preferred

def clip_boxes(box_array: np.ndarray, width: int, height: int) -> np.ndarray:
    box_array[:, [0, 2]] = np.clip(box_array[:, [0, 2]], 0, max(0, width - 1))
    box_array[:, [1, 3]] = np.clip(box_array[:, [1, 3]], 0, max(0, height - 1))
    return box_array

def sanitize_box(box: List[float], img_w: int, img_h: int, min_size: float = 2.0) -> List[float]:
    x1, y1, x2, y2 = map(float, box)
    x_min, x_max = sorted((x1, x2))
    y_min, y_max = sorted((y1, y2))

    x_min = max(0.0, min(x_min, img_w - 1 if img_w > 0 else 0))
    x_max = max(0.0, min(x_max, img_w - 1 if img_w > 0 else 0))
    y_min = max(0.0, min(y_min, img_h - 1 if img_h > 0 else 0))
    y_max = max(0.0, min(y_max, img_h - 1 if img_h > 0 else 0))

    if img_w > 1:
        min_w = min(min_size, max(1.0, float(img_w - 1)))
        if (x_max - x_min) < min_w:
            cx = (x_min + x_max) / 2.0
            half = min(min_w / 2.0, (img_w - 1) / 2.0)
            cx = min(max(cx, half), img_w - 1 - half)
            x_min = cx - half
            x_max = cx + half

    if img_h > 1:
        min_h = min(min_size, max(1.0, float(img_h - 1)))
        if (y_max - y_min) < min_h:
            cy = (y_min + y_max) / 2.0
            half = min(min_h / 2.0, (img_h - 1) / 2.0)
            cy = min(max(cy, half), img_h - 1 - half)
            y_min = cy - half
            y_max = cy + half

    x_min = max(0.0, min(x_min, img_w - 1 if img_w > 0 else 0))
    x_max = max(0.0, min(x_max, img_w - 1 if img_w > 0 else 0))
    y_min = max(0.0, min(y_min, img_h - 1 if img_h > 0 else 0))
    y_max = max(0.0, min(y_max, img_h - 1 if img_h > 0 else 0))

    return [x_min, y_min, x_max, y_max]


def draw_boxes(image: np.ndarray, boxes: List[List[float]], label: str = "leaf") -> np.ndarray:
    for x1, y1, x2, y2 in boxes:
        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)
        label_origin = (pt1[0], max(15, pt1[1] - 10))
        cv2.putText(image, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def boxes_to_coco_annotations(
    boxes: List[List[float]],
    image_id: int,
    next_annotation_id: int,
) -> tuple[List[dict], int]:
    annotations: List[dict] = []
    ann_id = next_annotation_id
    for x1, y1, x2, y2 in boxes:
        width = max(0.0, float(x2) - float(x1))
        height = max(0.0, float(y2) - float(y1))
        if width < 1.0 or height < 1.0:
            continue
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": CATEGORY_ID,
                "bbox": [float(x1), float(y1), width, height],
                "area": width * height,
                "iscrowd": 0,
                "segmentation": [],
            }
        )
        ann_id += 1
    return annotations, ann_id


def prepare_directory(path: Path, keep_existing: bool) -> None:
    if not keep_existing:
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def run_inference(
    model,
    img_path: Path,
    keywords: Sequence[str],
    caption: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> tuple[np.ndarray, List[List[float]]]:
    bgr_image = cv2.imread(str(img_path))
    if bgr_image is None:
        raise RuntimeError("Unreadable image")
    img_h, img_w = bgr_image.shape[:2]

    _, image_tensor = load_image(str(img_path))
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor[0],
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    boxes_np = np.asarray(boxes, dtype=np.float32)
    filtered_boxes: List[List[float]] = []
    if boxes_np.size > 0:
        scales = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
        boxes_abs = boxes_np * scales
        boxes_xyxy = np.empty_like(boxes_abs)
        boxes_xyxy[:, 0] = boxes_abs[:, 0] - boxes_abs[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes_abs[:, 1] - boxes_abs[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes_abs[:, 0] + boxes_abs[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes_abs[:, 1] + boxes_abs[:, 3] / 2.0
        boxes_xyxy = clip_boxes(boxes_xyxy, img_w, img_h)

        keyword_matches = 0
        lower_keywords = tuple(keyword.lower() for keyword in keywords)
        for box, phrase in zip(boxes_xyxy.tolist(), phrases):
            phrase_lower = phrase.lower()
            if any(keyword in phrase_lower for keyword in lower_keywords):
                filtered_boxes.append(box)
                keyword_matches += 1
        if keyword_matches == 0:
            filtered_boxes = boxes_xyxy.tolist()
            print("  - No keyword match in phrases, using all predicted boxes.")
        else:
            print(f"  - Filtered to {keyword_matches} box(es) matching keywords.")
    else:
        print("  - GroundingDINO produced zero boxes before keyword filtering.")

    normalized_boxes: List[List[float]] = []
    if filtered_boxes:
        normalized_boxes = [sanitize_box(box, img_w, img_h) for box in filtered_boxes]

    return bgr_image, normalized_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GroundingDINO labeling workflows.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Directory of input images.")
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR, help="Directory where COCO annotations are written.")
    parser.add_argument("--visualizations-dir", type=Path, default=DEFAULT_VISUALIZATIONS_DIR, help="Where visualization jpgs are written.")
    parser.add_argument("--visualize", action="store_true", help="Save debug images with drawn boxes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = resolve_device(PREFERRED_DEVICE)
    image_dir: Path = args.image_dir
    annotations_dir: Path = args.annotations_dir
    visualizations_dir: Path = args.visualizations_dir

    prepare_directory(annotations_dir, keep_existing=False)
    if args.visualize:
        prepare_directory(visualizations_dir, keep_existing=False)

    model = load_model(str(MODEL_CONFIG_PATH), str(MODEL_CHECKPOINT_PATH), device=device)

    image_paths = collect_images(image_dir)
    if not image_paths:
        print(f"No images found in {image_dir}.")
        return

    print(f"Processing {len(image_paths)} image(s) from {image_dir} on {device}.")

    images_meta: List[dict] = []
    annotations_meta: List[dict] = []
    next_annotation_id = 1

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] {img_path.name}")
        try:
            bgr_image, normalized_boxes = run_inference(
                model=model,
                img_path=img_path,
                keywords=KEYWORDS,
                caption=PROMPT_CAPTION,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=device,
            )
        except RuntimeError as exc:
            print(f"  - Skipping image: {exc}")
            continue

        img_h, img_w = bgr_image.shape[:2]
        images_meta.append(
            {
                "id": idx,
                "file_name": img_path.name,
                "width": img_w,
                "height": img_h,
            }
        )

        coco_annotations, next_annotation_id = boxes_to_coco_annotations(
            boxes=normalized_boxes,
            image_id=idx,
            next_annotation_id=next_annotation_id,
        )
        annotations_meta.extend(coco_annotations)
        print(f"  - Added {len(coco_annotations)} COCO box(es)")

        if args.visualize:
            vis_image = draw_boxes(bgr_image.copy(), normalized_boxes)
            vis_path = visualizations_dir / f"{img_path.stem}_boxed.jpg"
            cv2.imwrite(str(vis_path), vis_image)
            print(f"  - Saved visualization: {vis_path}")

    coco_dataset = {
        "images": images_meta,
        "annotations": annotations_meta,
        "categories": [
            {
                "id": CATEGORY_ID,
                "name": "leaf",
            }
        ],
    }
    annotations_path = annotations_dir / ANNOTATIONS_FILENAME
    annotations_path.write_text(json.dumps(coco_dataset, indent=2))
    print(
        f"Saved COCO annotations to {annotations_path} "
        f"({len(annotations_meta)} boxes across {len(images_meta)} image(s))."
    )
    print("Done.")


if __name__ == "__main__":
    main()
