#!/usr/bin/env python3
"""
Utility script for training a YOLOv8n model to detect Korean pepper leaves.

The script expects a dataset that follows the Ultralytics/YOLO darknet format:

dataset_root/
├── images/
│   ├── train/
│   ├── val/
│   └── test/               # optional
└── labels/
    ├── train/
    ├── val/
    └── test/               # optional

Each image must have a matching label file in the labels directory hierarchy,
using the YOLO bounding-box text format:

<class_id> <x_center> <y_center> <width> <height>

All coordinates are normalized to [0, 1] relative to the image size.

If you only have flat `images/` and `labels/` directories (no train/val
sub-folders), the script can create a random split automatically. Override the
`--val-fraction` flag to control the amount of data reserved for validation.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ultralytics import YOLO


def build_dataset_yaml(
    dataset_root: Path,
    output_dir: Path,
    class_name: str,
    yaml_name: Optional[str] = None,
) -> Path:
    """
    Create (or update) a dataset YAML file that YOLOv8 expects.

    Args:
        dataset_root: Root directory that contains the images/ and labels/ folders.
        output_dir: Directory where the YAML file should be written.
        class_name: Human-readable class name for index 0.
        yaml_name: Optional name for the YAML file. Defaults to <dataset_root.name>.yaml

    Returns:
        Path to the dataset YAML file.
    """
    dataset_root = dataset_root.resolve()
    yaml_name = yaml_name or f"{dataset_root.name}_pepper_leaf.yaml"
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


def validate_dataset_structure(dataset_root: Path) -> None:
    """Ensure the dataset directory contains the required YOLO structure."""
    expected_subdirs = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    ]

    missing = [str(path.relative_to(dataset_root)) for path in expected_subdirs if not path.exists()]
    if missing:
        missing_str = json.dumps(missing, indent=2)
        raise FileNotFoundError(
            f"Dataset root {dataset_root} is missing required directories:\n{missing_str}\n"
            "Expected YOLO layout with images/ and labels/ splits."
        )


def list_image_files(image_dir: Path) -> Sequence[Path]:
    """Return image files in a directory sorted by name."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in valid_exts])


def ensure_split_dataset(
    dataset_root: Path,
    scratch_dir: Path,
    val_fraction: float,
    seed: int,
    reuse_split: bool,
) -> Path:
    """
    Ensure the dataset has a YOLO-compliant train/val directory hierarchy.

    If the provided dataset already has the expected structure, the same root is
    returned. Otherwise the function creates a new split under scratch_dir.
    """
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    if (images_dir / "train").exists() and (labels_dir / "train").exists():
        return dataset_root

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Dataset at {dataset_root} must contain top-level 'images' and 'labels' directories."
        )

    if val_fraction <= 0 or val_fraction >= 1:
        raise ValueError("val_fraction must be between 0 and 1 (exclusive).")

    scratch_dir.mkdir(parents=True, exist_ok=True)
    split_name = f"{dataset_root.name}_split_v{int(val_fraction * 100):02d}_s{seed}"
    split_root = scratch_dir / split_name

    if split_root.exists():
        if reuse_split:
            print(f"[INFO] Reusing existing split dataset at {split_root}")
            return split_root
        print(f"[INFO] Removing existing split dataset at {split_root}")
        shutil.rmtree(split_root)

    image_files = list_image_files(images_dir)
    if not image_files:
        raise FileNotFoundError(f"No image files found in {images_dir}")

    rng = random.Random(seed)
    shuffled = image_files.copy()
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * val_fraction))
    val_set = set(shuffled[:val_count])

    # Prepare destination directories.
    for subdir in ("images/train", "images/val", "labels/train", "labels/val"):
        (split_root / subdir).mkdir(parents=True, exist_ok=True)

    train_count = 0
    val_count_actual = 0
    for img_path in image_files:
        label_src = labels_dir / f"{img_path.stem}.txt"
        if not label_src.exists():
            raise FileNotFoundError(f"Missing label file for {img_path.name} at {label_src}")

        if img_path in val_set:
            subset = "val"
            val_count_actual += 1
        else:
            subset = "train"
            train_count += 1

        img_dst = split_root / "images" / subset / img_path.name
        label_dst = split_root / "labels" / subset / f"{img_path.stem}.txt"
        shutil.copy2(img_path, img_dst)
        shutil.copy2(label_src, label_dst)

    print(
        f"[INFO] Created train/val split under {split_root} (train={train_count}, val={val_count_actual}, "
        f"val_fraction={val_fraction})"
    )
    return split_root


def coerce_value(value: str) -> Any:
    """Attempt to convert a CLI string value into an int, float, or bool."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_unknown_args(unknown: Sequence[str]) -> Dict[str, Any]:
    """Parse additional --key value pairs into a dictionary."""
    extras: Dict[str, Any] = {}
    idx = 0
    while idx < len(unknown):
        token = unknown[idx]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected argument format: {token}")
        key = token.lstrip("-").replace("-", "_")

        # Check if next token is a value or another flag.
        value: Any = True
        if idx + 1 < len(unknown) and not unknown[idx + 1].startswith("--"):
            value = coerce_value(unknown[idx + 1])
            idx += 1

        extras[key] = value
        idx += 1

    return extras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8n on Korean pepper leaf detections.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the dataset root containing images/ and labels/ folders.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Pretrained weights to start from (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size for training and validation.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/pepper_leaf"),
        help="Directory where training runs will be saved.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolov8n_pepper_leaf",
        help="Name of the training run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data-loading workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use (e.g., '0' for GPU id 0, 'cpu' for CPU; empty string lets Ultralytics decide).",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="Korean pepper leaf",
        help="Readable class name to store inside the dataset YAML file.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of images to allocate to the validation split when splitting automatically.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when generating automatic train/val splits.",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path("detection_yolo/.splits"),
        help="Directory where automatically generated dataset splits are stored.",
    )
    parser.add_argument(
        "--reuse-split",
        action="store_true",
        help="Reuse an existing automatic split directory if it already exists.",
    )
    args, unknown = parser.parse_known_args()
    args.yolo_overrides = parse_unknown_args(unknown)
    return args


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset_root
    output_dir: Path = Path(__file__).resolve().parent

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root {dataset_root} does not exist.")

    prepared_root = ensure_split_dataset(
        dataset_root=dataset_root,
        scratch_dir=args.scratch_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        reuse_split=args.reuse_split,
    )

    validate_dataset_structure(prepared_root)
    data_yaml = build_dataset_yaml(prepared_root, output_dir, args.class_name)

    model = YOLO(args.weights)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.img_size,
        "batch": args.batch,
        "project": str(args.project),
        "name": args.name,
        "workers": args.workers,
        "device": args.device,
    }
    train_kwargs.update(args.yolo_overrides)

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()