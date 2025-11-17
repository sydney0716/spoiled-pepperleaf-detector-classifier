"""
Resize and auto-orient all detection images into a fixed 640x640 canvas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageOps

DEFAULT_SIZE: Tuple[int, int] = (640, 640)
IMAGE_EXTENSIONS = {".jpg",}
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = PROJECT_ROOT / "data/interim/detection/selected_images"
TARGET_DIR = PROJECT_ROOT / "data/processed/detection/images"


def iter_images(root: Path) -> Iterable[Path]:
    """Yield image files under root (recursively)."""
    if not root.exists():
        raise FileNotFoundError(f"Input path not found: {root}")
    if root.is_file():
        if root.suffix.lower() in IMAGE_EXTENSIONS:
            yield root
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def resize_image(source: Path, destination: Path, size: Tuple[int, int]) -> None:
    """Resize source image to size and write to destination path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as img:
        oriented = ImageOps.exif_transpose(img)
        resized = oriented.resize(size, Image.BILINEAR)
        resized.save(destination, format=img.format or "PNG")


def resize_bulk(input_path: Path, output_path: Path, size: Tuple[int, int]) -> int:
    """
    Resize every image found under input_path into output_path, mirroring directories.
    """
    source_root = input_path if input_path.is_dir() else input_path.parent
    count = 0
    for img_path in iter_images(input_path):
        rel_path = img_path.relative_to(source_root)
        resize_image(img_path, output_path / rel_path, size)
        count += 1
    return count


def main() -> None:
    count = resize_bulk(SOURCE_DIR, TARGET_DIR, DEFAULT_SIZE)
    print(f"Resized {count} images from {SOURCE_DIR} to {TARGET_DIR} at {DEFAULT_SIZE}.")


if __name__ == "__main__":
    main()
