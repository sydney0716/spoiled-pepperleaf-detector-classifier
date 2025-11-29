from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS = PROJECT_ROOT / "models/trained/trained_resnet18.pth"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
CLASS_LABELS = ["normal", "spoiled"]


def build_model(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


def list_images(image_dir: Path) -> Sequence[Path]:
    return sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_checkpoint(weights_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(weights_path, map_location=device)
    model = build_model()
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the trained ResNet18 pepper leaf classifier on a directory of images."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing leaf crops to classify.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"Path to the trained checkpoint (default: {DEFAULT_WEIGHTS}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use (default: auto-detect).",
    )
    return parser.parse_args()


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    weights_path = args.weights.resolve()
    device = torch.device(args.device)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    images = list_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No supported images found under {input_dir}")

    model = load_checkpoint(weights_path, device)
    transform = build_transform()

    print(f"[INFO] Running classification on {len(images)} images using {weights_path.name}")
    with torch.inference_mode():
        for img_path in images:
            image = Image.open(img_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)
            label = CLASS_LABELS[pred_idx.item()]
            print(f"{img_path.name}: {label} ({conf.item():.2%})")


if __name__ == "__main__":
    main()
