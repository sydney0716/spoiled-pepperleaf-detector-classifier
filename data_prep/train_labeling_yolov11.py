"""
This script trains YOLOv11 using an existing YOLO-formatted dataset.
Reads dataset config from data/labeling_data_for_labeling_yolo11s/dataset.yaml.
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
DEFAULT_DATASET_PATH = PROJECT_ROOT / "detection_yolo/.splits/detection_processed_split/"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "runs/labeling"

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


def train_with_fallback(train_kwargs: Dict[str, object], dataset_path: Path) -> None:
    model_order = ["yolo11s.pt", "yolo11n.pt"]
    last_error = None

    for model_name in model_order:
        try:
            print(f"\n=== Attempting training with {model_name} ===")
            model = YOLO(model_name)
            model_kwargs = dict(train_kwargs)
            model_kwargs["data"] = str(dataset_path)
            model.train(**model_kwargs)
            print(f"\nTraining completed successfully with {model_name}.")
            return
        except RuntimeError as exc:
            error_text = str(exc)
            last_error = exc
            if "out of memory" in error_text.lower() and model_name != model_order[-1]:
                print(f"\nCUDA OOM encountered with {model_name}. Clearing cache and trying fallback model...")
                torch.cuda.empty_cache()
                continue
            raise

    if last_error is not None:
        raise last_error


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
        train_with_fallback(train_kwargs, dataset_path)
    except Exception as exc:  # noqa: BLE001
        print(f"\nTraining failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()