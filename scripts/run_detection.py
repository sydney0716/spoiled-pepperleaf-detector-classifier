from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "models/trained/trained_yolov8n.pt"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "runs"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(image_dir: Path) -> Sequence[Path]:
    """Return sorted image paths that YOLO can ingest."""
    return sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8n detection using the trained pepper leaf checkpoint."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory containing the images to analyze.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help=f"Directory where dection results will be saved (default: {DEFAULT_SAVE_DIR}).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"Path to the YOLO model weights (default: {DEFAULT_WEIGHTS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    weights_path = args.weights.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

    images = list_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No supported image files found under {input_dir}.")

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Expected trained weights at {weights_path}. Please provide a valid path."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{weights_path.stem}_{input_dir.name}"

    model = YOLO(str(weights_path))
    model.predict(
        source=str(input_dir),
        project=str(output_dir),
        name=run_name,
        imgsz=640,
        conf=0.25,
        save=True,
        save_txt=False,  # draw boxes only; skip label/score overlays and text exports
        show_labels=False,
        show_conf=False,
        exist_ok=True,
    )

    print(f"[INFO] Processed {len(images)} images from {input_dir}")
    print(f"[INFO] YOLO outputs saved under {output_dir / run_name}")


if __name__ == "__main__":
    main()