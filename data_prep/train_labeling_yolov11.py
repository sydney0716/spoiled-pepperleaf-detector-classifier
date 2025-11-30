"""
This script trains YOLOv11 using an existing YOLO-formatted dataset.
Reads dataset config from data/detection_processed/detection_processed_split.yaml.
Saves training results to runs/labeling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data/detection_processed/detection_processed_split.yaml"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "runs/labeling"
PRETRAINED_MODEL_PATH = PROJECT_ROOT / "models/pretrained/yolo11s.pt"

DEFAULT_EPOCHS = 150
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 8
DEFAULT_DEVICE = "0"
DEFAULT_NAME = "labeling_yolov11"
DEFAULT_PATIENCE = 50
DEFAULT_HALF = False
DEFAULT_ACCUMULATE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 using existing YOLO formatted data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the directory containing the YOLO dataset.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help="Destination directory where the training results should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.data_dir.resolve()

    save_dir = args.save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs: Dict[str, object] = dict(
        data=dataset_path,
        epochs=DEFAULT_EPOCHS,
        imgsz=DEFAULT_IMGSZ,
        batch=DEFAULT_BATCH,
        device=DEFAULT_DEVICE,
        project=str(save_dir),
        name=DEFAULT_NAME,
        patience=DEFAULT_PATIENCE,
        half=DEFAULT_HALF,
    )
    if DEFAULT_ACCUMULATE and DEFAULT_ACCUMULATE > 1:
        train_kwargs["accumulate"] = DEFAULT_ACCUMULATE

    try:
        if not PRETRAINED_MODEL_PATH.exists():
            raise FileNotFoundError(f"Pretrained model not found: {PRETRAINED_MODEL_PATH}")

        model = YOLO(str(PRETRAINED_MODEL_PATH))
        model.train(**train_kwargs)
        print(f"\nTraining completed successfully with yolo11s.pt.")
    except Exception as exc:  # noqa: BLE001
        print(f"\nTraining failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()