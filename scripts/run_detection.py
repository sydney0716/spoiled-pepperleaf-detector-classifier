from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS = PROJECT_ROOT / "models/trained/trained_yolov8n.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs"
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
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the images to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where YOLO run artifacts will be saved (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

    images = list_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No supported image files found under {input_dir}.")

    if not DEFAULT_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Expected trained weights at {DEFAULT_WEIGHTS}. Run train_detection_yolo.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"yolov8n_{input_dir.name}"

    model = YOLO(str(DEFAULT_WEIGHTS))
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
