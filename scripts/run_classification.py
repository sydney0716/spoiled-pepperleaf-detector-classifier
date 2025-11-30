from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHT_PATH = PROJECT_ROOT / "models/trained/trained_resnet18.pth"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "runs/classification"
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

def build_model(model_name: str = "resnet18", num_classes: int = 2) -> nn.Module:
    if model_name not in ARCH_CONFIG:
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {list(ARCH_CONFIG.keys())}")

    config = ARCH_CONFIG[model_name]
    # In inference/production, we don't strictly need 'pretrained=True' if we are loading weights,
    # but creating the structure is necessary.
    model = config["builder"](weights=None)

    # We reconstruct the same modification to fc layer as during training
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


def list_images(image_dir: Path) -> Sequence[Path]:
    return sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_checkpoint(model_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(model_path, map_location=device)
    
    model_name = checkpoint.get("model_name", checkpoint.get("backbone", "resnet18"))
    
    print(f"[INFO] Detected model architecture: {model_name}")
    
    model = build_model(model_name=model_name)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the trained pepper leaf classifier on a directory of images."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory containing leaf crops to classify.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHT_PATH,
        help=f"Path to the trained model checkpoint (default: {DEFAULT_WEIGHT_PATH}).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help=f"Directory to save classification results (default: {DEFAULT_SAVE_DIR}).",
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
    source_dir = args.source.resolve()
    model_path = args.model.resolve()
    save_dir = args.save_dir.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Auto-detect device

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    images = list_images(source_dir)
    if not images:
        raise FileNotFoundError(f"No supported images found under {source_dir}")

    save_dir.mkdir(parents=True, exist_ok=True)
    results_file = save_dir / "classification_results.txt"

    model = load_checkpoint(model_path, device)
    transform = build_transform()

    print(f"[INFO] Running classification on {len(images)} images using {model_path.name}")
    
    classification_outputs = []
    with torch.inference_mode():
        for img_path in images:
            image = Image.open(img_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)
            label = CLASS_LABELS[pred_idx.item()]
            
            output_line = f"{img_path.name}: {label} ({conf.item():.2%})"
            print(output_line)
            classification_outputs.append(output_line)
            
    with open(results_file, "w") as f:
        for line in classification_outputs:
            f.write(line + "\n")
    print(f"[INFO] Classification results saved to {results_file}")


if __name__ == "__main__":
    main()