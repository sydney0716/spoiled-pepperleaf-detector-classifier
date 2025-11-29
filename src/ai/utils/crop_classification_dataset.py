#!/usr/bin/env python3
"""
Crop pepper-leaf classification images using JSON annotations and resize to 224x224.

Input layout (source):
  data/classification/classification_raw/
    images/
      normal/*.jpg
      spoiled/<disease>/*.jpg
    labels/
      normal/*.json
      spoiled/<disease>/*.json

Output layout (destination):
  data/classification/classification_processed/
    images/
      normal/*.jpg
      spoiled/<disease>/*.jpg
    labels/  (JSON files copied verbatim)
      normal/*.json
      spoiled/<disease>/*.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SRC = PROJECT_ROOT / "data/classification/classification_raw"
DEFAULT_DST = PROJECT_ROOT / "data/classification/classification_processed"
OUTPUT_SIZE = (224, 224)


def read_bbox(label_path: Path) -> Optional[Tuple[int, int, int, int]]:
    """Return (xtl, ytl, xbr, ybr) from the first annotation point, if present."""
    data = json.loads(label_path.read_text())
    points: Iterable[dict] = data.get("annotations", {}).get("points") or []
    if not points:
        return None
    p = next(iter(points))
    try:
        return int(p["xtl"]), int(p["ytl"]), int(p["xbr"]), int(p["ybr"])
    except Exception:  # noqa: BLE001
        return None


def ensure_dirs(dst_root: Path, subpath: Path) -> Path:
    out_dir = dst_root / subpath
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def crop_and_resize(img_path: Path, bbox: Optional[Tuple[int, int, int, int]]) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    if bbox:
        xtl, ytl, xbr, ybr = bbox
        xtl = max(0, xtl)
        ytl = max(0, ytl)
        xbr = min(img.width, xbr)
        ybr = min(img.height, ybr)
        if xtl < xbr and ytl < ybr:
            img = img.crop((xtl, ytl, xbr, ybr))
    return img.resize(OUTPUT_SIZE)


def process_split(src_root: Path, dst_root: Path, split: str) -> None:
    src_images = src_root / "images" / split
    src_labels = src_root / "labels" / split
    if not src_images.exists() or not src_labels.exists():
        print(f"[WARN] Missing split '{split}' in {src_root}, skipping.")
        return

    # For spoiled: label JSONs live under labels/spoiled/<disease>/...
    # For normal: label JSONs live directly under labels/normal.
    potential_classes = []
    for entry in sorted(src_labels.iterdir()):
        if entry.is_dir():
            potential_classes.append(entry)
    if not potential_classes:
        potential_classes.append(src_labels)  # handle flat JSONs (e.g., normal)

    for class_dir in potential_classes:
        class_name = class_dir.name if class_dir.is_dir() else split
        rel_class = Path(split) / class_name
        # Normal images live directly under images/normal; spoiled under images/spoiled/<disease>
        if split == "normal":
            rel_class = Path("normal")
        dst_img_dir = ensure_dirs(dst_root / "images", rel_class)

        label_files = sorted(class_dir.glob("*.json")) if class_dir.is_dir() else sorted(src_labels.glob("*.json"))
        if not label_files:
            print(f"[WARN] No labels found in {class_dir}")
            continue

        for idx, label_file in enumerate(label_files, start=1):
            label_content = label_file.read_text()
            label_json = json.loads(label_content)
            bbox = read_bbox(label_file)
            src_img_name = label_json.get("description", {}).get("image") or label_file.stem.replace(".json", "")
            # Normal split stores images directly under images/normal; spoiled uses images/spoiled/<disease>/...
            if class_name == split:
                src_img = src_images / src_img_name
            else:
                src_img = src_images / class_name / src_img_name
            if not src_img.exists():
                print(f"[WARN] Image missing for label {label_file}, expected {src_img}")
                continue

            try:
                cropped = crop_and_resize(src_img, bbox)
                cropped.save(dst_img_dir / src_img.name)
                if idx % 10 == 0:
                    print(f"[INFO] Processed {idx} images in class '{rel_class}'")
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Failed to process {src_img}: {exc}")


def main() -> None:
    src_root = DEFAULT_SRC
    dst_root = DEFAULT_DST
    for split in ("normal", "spoiled"):
        process_split(src_root, dst_root, split)
    print(f"[DONE] Cropped images saved to {dst_root}")


if __name__ == "__main__":
    main()
