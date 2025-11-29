from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_COCO_JSON = REPO_ROOT / "data/labeling_data_for_labeling_yolo11s/_annotations.coco.json"
DEFAULT_IMAGE_ROOT = REPO_ROOT / "data/detection_processed/images"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs/labeled_data_yolo11s"
DEFAULT_PROJECT_ROOT = REPO_ROOT / "results/labeling_yolo11s"


DEFAULT_CLASS_NAME = "leaf"
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SPLIT_SEED = 42
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 8
DEFAULT_DEVICE = "0"
DEFAULT_NAME = "train_yolov11"
DEFAULT_PATIENCE = 50
DEFAULT_HALF = False
DEFAULT_ACCUMULATE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 using the hand-reviewed COCO annotations. Only directory locations are configurable."
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Directory that stores the resized pepper-leaf images.",
    )
    parser.add_argument(                                                                      
        "--output-dir",                                                                       
        type=Path,                                                                            
        default=DEFAULT_OUTPUT_ROOT,                                                          
        help="Destination directory where the YOLO-formatted output should be written.",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="Directory where Ultralytics should write its training runs.",
    )
    return parser.parse_args()


def resolve_image_path(image_info: Dict, image_root: Path) -> Path:
    candidate = image_root / Path(image_info["file_name"]).name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find an image for COCO entry id={image_info['id']} under {image_root}."
    )


def build_category_mapping(coco_dict: Dict) -> Tuple[Dict[int, int], List[str]]:
    used_category_ids = sorted({ann["category_id"] for ann in coco_dict["annotations"]})
    if not used_category_ids:
        raise ValueError("COCO file does not contain any annotations.")
    id_to_name = {cat["id"]: cat.get("name", str(cat["id"])) for cat in coco_dict["categories"]}
    mapping: Dict[int, int] = {}
    class_names: List[str] = []
    for category_id in used_category_ids:
        mapping[category_id] = len(class_names)
        class_names.append(id_to_name.get(category_id, str(category_id)))
    return mapping, class_names


def convert_bbox_to_yolo(
    bbox: Iterable[float], width: float, height: float, class_idx: int
) -> str:
    x, y, w, h = bbox
    x_c = (x + w / 2.0) / width
    y_c = (y + h / 2.0) / height
    w_n = w / width
    h_n = h / height
    return f"{class_idx} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"


def split_image_ids(image_ids: List[int], val_fraction: float, seed: int) -> Tuple[set[int], set[int]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")
    if len(image_ids) < 2:
        raise ValueError("Need at least two images to create a train/val split.")
    rng = random.Random(seed)
    shuffled = image_ids[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_fraction))
    val_ids = set(shuffled[:val_count])
    train_ids = set(shuffled[val_count:])
    if not train_ids:
        # Guarantee at least one training example if the dataset is tiny.
        train_ids.add(shuffled[val_count - 1])
        val_ids.remove(shuffled[val_count - 1])
    return train_ids, val_ids


def ensure_dir_structure(root: Path) -> None:
    for relative in ("images/train", "images/val", "labels/train", "labels/val"):
        (root / relative).mkdir(parents=True, exist_ok=True)


def save_label_file(label_path: Path, lines: List[str]) -> None:
    if lines:
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        label_path.write_text("", encoding="utf-8")


def build_dataset_yaml(dataset_root: Path, class_names: List[str]) -> Path:
    yaml_path = dataset_root / "dataset.yaml"
    lines = [
        f"path: {dataset_root}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def prepare_dataset_from_coco(
    coco_json: Path,
    image_root: Path,
    dataset_out: Path,
    val_fraction: float,
    seed: int,
    class_name_override: str | None,
) -> Path:
    if not coco_json.exists():
        raise FileNotFoundError(f"COCO file not found: {coco_json}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    coco_dict = json.loads(coco_json.read_text())
    category_mapping, class_names = build_category_mapping(coco_dict)
    if class_name_override and len(class_names) == 1:
        class_names[0] = class_name_override

    image_map = {img["id"]: img for img in coco_dict["images"]}
    annotations_by_image: Dict[int, List[str]] = {img_id: [] for img_id in image_map}

    for ann in coco_dict["annotations"]:
        image_id = ann["image_id"]
        if image_id not in image_map:
            continue
        img_info = image_map[image_id]
        class_idx = category_mapping[ann["category_id"]]
        label_line = convert_bbox_to_yolo(ann["bbox"], img_info["width"], img_info["height"], class_idx)
        annotations_by_image[image_id].append(label_line)

    dataset_root = dataset_out.resolve()
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    ensure_dir_structure(dataset_root)

    train_ids, val_ids = split_image_ids(list(image_map.keys()), val_fraction, seed)
    split_lookup = {"train": train_ids, "val": val_ids}

    for split_name, split_ids in split_lookup.items():
        for image_id in split_ids:
            img_info = image_map[image_id]
            src_img = resolve_image_path(img_info, image_root)
            dst_img = dataset_root / "images" / split_name / src_img.name
            shutil.copy2(src_img, dst_img)
            dst_label = dataset_root / "labels" / split_name / f"{dst_img.stem}.txt"
            save_label_file(dst_label, annotations_by_image.get(image_id, []))

    yaml_path = build_dataset_yaml(dataset_root, class_names)
    print(
        f"Prepared YOLO dataset at {dataset_root} "
        f"(train={len(train_ids)}, val={len(val_ids)}, classes={class_names})"
    )
    return yaml_path


def train_with_fallback(train_kwargs: Dict[str, object]) -> None:
    model_order = ["yolo11s.pt", "yolo11n.pt"]
    last_error = None

    for model_name in model_order:
        try:
            print(f"\n=== Attempting training with {model_name} ===")
            model = YOLO(model_name)
            model_kwargs = dict(train_kwargs)
            model_kwargs["data"] = str(model_kwargs["data"])
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
    coco_json = DEFAULT_COCO_JSON
    if not coco_json.exists():
        raise FileNotFoundError(f"Required COCO annotations not found at {coco_json}")

    image_root = (args.image_root or DEFAULT_IMAGE_ROOT).resolve()
    if not image_root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    dataset_root = args.output_dir.resolve()
    project_dir = (args.project_dir or DEFAULT_PROJECT_ROOT).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = prepare_dataset_from_coco(
        coco_json=coco_json,
        image_root=image_root,
        dataset_out=dataset_root,
        val_fraction=DEFAULT_VAL_FRACTION,
        seed=DEFAULT_SPLIT_SEED,
        class_name_override=DEFAULT_CLASS_NAME,
    )

    train_kwargs: Dict[str, object] = dict(
        data=data_yaml,
        epochs=DEFAULT_EPOCHS,
        imgsz=DEFAULT_IMGSZ,
        batch=DEFAULT_BATCH,
        device=DEFAULT_DEVICE,
        project=str(project_dir),
        name=DEFAULT_NAME,
        patience=DEFAULT_PATIENCE,
        half=DEFAULT_HALF,
    )
    if DEFAULT_ACCUMULATE and DEFAULT_ACCUMULATE > 1:
        train_kwargs["accumulate"] = DEFAULT_ACCUMULATE

    try:
        train_with_fallback(train_kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"\nTraining failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
