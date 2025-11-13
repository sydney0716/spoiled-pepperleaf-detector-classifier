#!/usr/bin/env python3
"""
Train YOLOv8n on Korean pepper leaves using the flat dataset in data/processed/detection.

The dataset only contains `images/` and `labels/`. This script automatically creates a
deterministic train/val split under an intermediate directory, generates the required
dataset YAML, and launches Ultralytics training with fixed hyper-parameters.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Optional, Sequence

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data/processed/detection"
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights/pretrained/yolov8n.pt"
DEFAULT_EPOCHS = 150
DEFAULT_BATCH = 16
DEFAULT_IMG_SIZE = 640
DEFAULT_PROJECT = PROJECT_ROOT / "results/detection"
DEFAULT_RUN_NAME = "yolov8n"
DEFAULT_WORKERS = 8
DEFAULT_DEVICE = ""
DEFAULT_CLASS_NAME = "Pepper leaf"
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SEED = 42
DEFAULT_SCRATCH_DIR = PROJECT_ROOT / "detection_yolo/.splits"


def build_dataset_yaml(
    dataset_root: Path,
    output_dir: Path,
    class_name: str,
    yaml_name: Optional[str] = None,
) -> Path:

    dataset_root = dataset_root.resolve()
    yaml_name = yaml_name or f"{dataset_root.name}.yaml"
    yaml_path = output_dir / yaml_name

    data_config = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test" if (dataset_root / "images" / "test").exists() else None,
        "names": {0: class_name},
    }

    # Remove optional entries that are not present to keep the YAML minimal.
    if data_config["test"] is None:
        data_config.pop("test")

    yaml_content_lines = [
        f"path: {data_config['path']}",
        f"train: {data_config['train']}",
        f"val: {data_config['val']}",
        "names:",
        f"  0: {class_name}",
    ]

    if "test" in data_config:
        # Insert the test line after the val entry
        yaml_content_lines.insert(3, f"test: {data_config['test']}")

    yaml_path.write_text("\n".join(yaml_content_lines) + "\n", encoding="utf-8")
    return yaml_path


def list_image_files(image_dir: Path) -> Sequence[Path]:
    """Return image files in a directory sorted by name."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in valid_exts])


def prepare_split_dataset(dataset_root: Path, scratch_dir: Path, val_fraction: float, seed: int) -> Path:
    """
    Create a YOLO-compliant train/val split from a flat images/labels dataset.
    """
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Expected `images/` and `labels/` directories under {dataset_root}, but at least one is missing."
        )

    image_files = list_image_files(images_dir)
    if not image_files:
        raise FileNotFoundError(f"No image files found in {images_dir}")

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1 (exclusive).")

    scratch_dir = scratch_dir.resolve()
    scratch_dir.mkdir(parents=True, exist_ok=True)
    split_root = scratch_dir / f"{dataset_root.name}_split"
    if split_root.exists():
        shutil.rmtree(split_root)

    for subdir in ("images/train", "images/val", "labels/train", "labels/val"):
        (split_root / subdir).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    shuffled = image_files.copy()
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_fraction))
    val_set = set(shuffled[:val_count])

    train_count = 0
    val_count_actual = 0
    for img_path in image_files:
        label_src = labels_dir / f"{img_path.stem}.txt"
        if not label_src.exists():
            raise FileNotFoundError(f"Missing label file for {img_path.name} at {label_src}")

        subset = "val" if img_path in val_set else "train"
        if subset == "val":
            val_count_actual += 1
        else:
            train_count += 1

        img_dst = split_root / "images" / subset / img_path.name
        label_dst = split_root / "labels" / subset / f"{img_path.stem}.txt"
        shutil.copy2(img_path, img_dst)
        shutil.copy2(label_src, label_dst)

    print(
        f"[INFO] Prepared split at {split_root} (train={train_count}, val={val_count_actual}, "
        f"val_fraction={val_fraction}, seed={seed})"
    )
    return split_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n on Korean pepper leaf detections with fixed hyper-parameters."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=f"Path to the dataset root containing images/ and labels/ folders (default: {DEFAULT_DATASET_ROOT}).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=f"Optional path to pretrained weights (default: {DEFAULT_WEIGHTS}).",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help=f"Directory where training runs will be saved (default: {DEFAULT_PROJECT}).",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help=f"Directory where automatic splits are cached (default: {DEFAULT_SCRATCH_DIR}).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root: Path = (args.dataset_root or DEFAULT_DATASET_ROOT).resolve()
    output_dir: Path = Path(__file__).resolve().parent

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root {dataset_root} does not exist.")

    scratch_dir = (args.scratch_dir or DEFAULT_SCRATCH_DIR).resolve()
    split_root = prepare_split_dataset(
        dataset_root=dataset_root,
        scratch_dir=scratch_dir,
        val_fraction=DEFAULT_VAL_FRACTION,
        seed=DEFAULT_SEED,
    )

    data_yaml = build_dataset_yaml(split_root, output_dir, DEFAULT_CLASS_NAME)

    weights_path = (args.weights or DEFAULT_WEIGHTS)
    project_dir = (args.project_dir or DEFAULT_PROJECT).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": DEFAULT_EPOCHS,
        "imgsz": DEFAULT_IMG_SIZE,
        "batch": DEFAULT_BATCH,
        "project": str(project_dir),
        "name": DEFAULT_RUN_NAME,
        "workers": DEFAULT_WORKERS,
        "device": DEFAULT_DEVICE,
    }

    model.train(**train_kwargs)
    trainer = getattr(model, "trainer", None)
    best_weight_path = None
    if trainer is not None:
        best_attr = getattr(trainer, "best", None)
        if best_attr:
            best_weight_path = Path(best_attr)
    if best_weight_path and best_weight_path.exists():
        trained_dir = PROJECT_ROOT / "weights/trained"
        trained_dir.mkdir(parents=True, exist_ok=True)
        weight_name = Path(weights_path).stem
        target_path = trained_dir / f"best_{weight_name}.pt"
        shutil.copy2(best_weight_path, target_path)
        print(f"[INFO] Copied best weights to {target_path}")


if __name__ == "__main__":
    main()
