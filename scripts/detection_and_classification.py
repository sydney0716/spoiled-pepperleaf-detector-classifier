from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DET_WEIGHTS = PROJECT_ROOT / "models/trained/trained_yolov8s.pt"
DEFAULT_CLS_WEIGHTS = PROJECT_ROOT / "models/trained/trained_resnet18.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs/pipeline"
DEFAULT_DET_CONF = 0.25
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
CLASS_LABELS = ["normal", "spoiled"]

ARCH_CONFIG = {
    "resnet18": {
        "builder": models.resnet18,
        "hidden_features": 256,
    },
    "resnet50": {
        "builder": models.resnet50,
        "hidden_features": 512,
    },
}

def is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def build_classifier(model_name: str = "resnet18", num_classes: int = 2) -> nn.Module:
    if model_name not in ARCH_CONFIG:
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {list(ARCH_CONFIG.keys())}")

    config = ARCH_CONFIG[model_name]
    model = config["builder"](weights=None)
    
    # Freeze/Unfreeze logic same as training (optional for inference but good for matching structure)
    # Actually for inference we just need the architecture structure to load state_dict
    
    num_features = model.fc.in_features
    hidden = config["hidden_features"]
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, num_classes),
    )
    return model


def load_classifier(weights_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(weights_path, map_location=device)
    
    model_name = checkpoint.get("model_name", checkpoint.get("backbone", "resnet18"))
    print(f"[INFO] Detected classifier architecture: {model_name}")
    
    model = build_classifier(model_name=model_name)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[int, int, int, int]:
    return (
        max(0, int(x1)),
        max(0, int(y1)),
        min(width, int(x2)),
        min(height, int(y2)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run detection first, then classify each detected leaf crop."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Single image to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write annotated results (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--det-weights",
        type=Path,
        default=DEFAULT_DET_WEIGHTS,
        help=f"Path to detection weights (default: {DEFAULT_DET_WEIGHTS}).",
    )
    parser.add_argument(
        "--cls-weights",
        type=Path,
        default=DEFAULT_CLS_WEIGHTS,
        help=f"Path to classification weights (default: {DEFAULT_CLS_WEIGHTS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()
    overlay_dir = output_dir
    
    det_weights = args.det_weights.resolve()
    cls_weights = args.cls_weights.resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not is_supported_image(image_path):
        raise FileNotFoundError(f"Unsupported image type: {image_path.suffix}")
    if not det_weights.exists():
        raise FileNotFoundError(f"Detection checkpoint not found: {det_weights}")
    if not cls_weights.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {cls_weights}")

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Processing image {image_path.name}")
    print(f"[INFO] Using detection weights: {det_weights.name}")
    print(f"[INFO] Using classification weights: {cls_weights.name}")
    
    det_model = YOLO(str(det_weights))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_model = load_classifier(cls_weights, device)
    transform = build_transform()
    font = ImageFont.load_default()

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    det_results = det_model.predict(
        source=str(image_path),
        imgsz=640,
        conf=DEFAULT_DET_CONF,
        verbose=False,
        save=False,
    )
    result = det_results[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        print(f"[WARN] No detections for {image_path.name}; writing original image.")
        image.save(overlay_dir / image_path.name)
        return

    draw = ImageDraw.Draw(image)
    with torch.inference_mode():
        for det_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, width, height)

            crop = image.crop((x1, y1, x2, y2))
            tensor = transform(crop).unsqueeze(0).to(device)
            logits = cls_model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)
            label = CLASS_LABELS[pred_idx.item()]
            conf_pct = conf.item() * 100.0

            color = (0, 200, 0) if label == "normal" else (220, 0, 0)
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)

            text = f"{label} {conf_pct:.1f}%"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            text_bg = [x1, max(0, y1 - text_h - 4), x1 + text_w + 8, y1]
            draw.rectangle(text_bg, fill=color)
            draw.text((text_bg[0] + 4, text_bg[1] + 2), text, fill="white", font=font)

    overlay_path = overlay_dir / image_path.name
    image.save(overlay_path)
    print(f"[INFO] {image_path.name}: {len(boxes)} detections -> {overlay_path.relative_to(PROJECT_ROOT)}")

    print(f"[INFO] Done. Overlays saved to {overlay_dir}")


if __name__ == "__main__":
    main()