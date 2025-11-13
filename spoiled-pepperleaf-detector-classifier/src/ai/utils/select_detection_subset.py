#!/usr/bin/env python3
"""
Copy the curated detection subset from the raw dataset into the processed folder.

This lightweight helper only deals with `.jpg` images: given a text file that
lists the original basenames (without extensions) it looks up the matching raw
image under `data/raw/detection_raw/images` and copies it into
`data/interim/detection/selected_image`. Run it whenever you need to refresh the
curated subset without manually dragging files around.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

RepoLookups = Tuple[Dict[str, Path], Dict[str, Path]]

IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg",)


@dataclass
class CopyStats:
    copied: int = 0
    missing: List[str] = field(default_factory=list)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_paths() -> dict[str, Path]:
    root = repo_root()
    data_dir = root / "data"
    raw = data_dir / "raw" / "detection_raw"
    interim = data_dir / "interim" / "detection" / "selected_images"
    return {
        "selected_list": interim / "selected_images.txt",
        "raw_images": raw / "imgs",
        "out_images": interim,
    }


def parse_args(defaults: dict[str, Path]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the curated detection subset from raw into processed directories."
    )
    parser.add_argument(
        "--raw-images",
        type=Path,
        default=defaults["raw_images"],
        help="Directory that contains the raw detection images.",
    )
    parser.add_argument(
        "--out-images",
        type=Path,
        default=defaults["out_images"],
        help="Destination directory for the curated images.",
    )
    return parser.parse_args()


def load_selection(list_path: Path) -> List[str]:
    if not list_path.exists():
        raise FileNotFoundError(f"Selected list not found: {list_path}")
    names: List[str] = []
    with list_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line)
    if not names:
        raise ValueError(f"No filenames found in {list_path}")
    return names


def build_lookups(directory: Path) -> RepoLookups:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    by_name: Dict[str, Path] = {}
    by_stem: Dict[str, Path] = {}
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        lower_name = entry.name.lower()
        by_name[lower_name] = entry
        by_stem.setdefault(entry.stem.lower(), entry)
    if not by_name:
        raise FileNotFoundError(f"No files found inside {directory}")
    return by_name, by_stem


def resolve_path(
    target: str,
    lookups: RepoLookups,
    extensions: Sequence[str],
) -> Path:
    by_name, by_stem = lookups
    has_suffix = Path(target).suffix != ""
    candidates = [target]
    if not has_suffix:
        candidates.extend(f"{target}{ext}" for ext in extensions)
    for candidate in candidates:
        match = by_name.get(candidate.lower())
        if match:
            return match
    if not has_suffix:
        stem_match = by_stem.get(target.lower())
        if stem_match:
            return stem_match
    raise FileNotFoundError(f"Could not find a match for '{target}'")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    ensure_directory(dst.parent)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)


def copy_subset(raw_dir: Path, out_dir: Path, selected_list: Path) -> CopyStats:
    names = load_selection(selected_list)
    image_lookup = build_lookups(raw_dir)
    stats = CopyStats()

    for name in names:
        try:
            image_src = resolve_path(name, image_lookup, IMAGE_EXTENSIONS)
        except FileNotFoundError:
            stats.missing.append(name)
            print(f"Image not found for '{name}'")
            continue

        image_dst = out_dir / image_src.name
        copy_file(image_src, image_dst)
        stats.copied += 1

    return stats


def summarize(stats: CopyStats, dest_images: Path) -> None:
    print(f"Images copied: {stats.copied} -> {dest_images}")
    if stats.missing:
        missing = ", ".join(stats.missing[:5])
        extra = "" if len(stats.missing) <= 5 else f" (+{len(stats.missing) - 5} more)"
        print(f"Missing images for: {missing}{extra}")


def main() -> None:
    defaults = default_paths()
    args = parse_args(defaults)
    ensure_directory(args.out_images)
    stats = copy_subset(args.raw_images, args.out_images, defaults["selected_list"])
    summarize(stats, args.out_images)
    if stats.missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
