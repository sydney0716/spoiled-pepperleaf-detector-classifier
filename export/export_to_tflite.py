#!/usr/bin/env python3
"""
Convert trained pepper leaf models into Raspberry Pi ready artifacts.

The script searches the training output folders for YOLO detection weights
(`*.pt`) and ResNet classification checkpoints (`*.pth`) and exports them to
TFLite for Raspberry Pi deployment. All outputs land in `weights/exported`
(override with `--export-dir` for classification exports).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO
s
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINED_ROOT = PROJECT_ROOT / "models" / "trained"
EXPORT_ROOT = PROJECT_ROOT / "models" / "exported"
DEFAULT_PATHS: Dict[str, Path] = {
    "yolo": TRAINED_ROOT, # Renamed from "detection"
    "resnet": TRAINED_ROOT, # Renamed from "classification"
    "export": EXPORT_ROOT,
}

CLASSIFIER_ARCH = {
    "resnet18": {"builder": models.resnet18, "hidden_features": 256},
    "resnet50": {"builder": models.resnet50, "hidden_features": 512},
}

def create_classifier(model_name: str, num_classes: int = 2) -> nn.Module:
    if model_name not in CLASSIFIER_ARCH:
        raise ValueError(f"Unsupported model '{model_name}'. Expected one of {tuple(CLASSIFIER_ARCH)}.")
    config = CLASSIFIER_ARCH[model_name]
    model = config["builder"](weights=None)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export trained detection/classification models for Raspberry Pi deployment."
    )
    parser.add_argument(
        "--yolo-dir",
        type=Path,
        default=DEFAULT_PATHS["yolo"],
        help="Directory that stores YOLO detection checkpoints (*.pt).",
    )
    parser.add_argument(
        "--resnet-dir",
        type=Path,
        default=DEFAULT_PATHS["resnet"], 
        help="Directory that stores ResNet classification checkpoints (*.pth).",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_PATHS["export"],
        help="Directory where Raspberry Pi compatible artifacts will be written.",
    )
    return parser.parse_args()


def list_weight_files(directory: Path, suffix: str) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob(f"*{suffix}") if p.is_file())


def export_detection_model(weights_path: Path, export_dir: Path) -> List[Path]:
    model = YOLO(str(weights_path))
    run_name = f"{weights_path.stem}_tflite"
    print(f"[DETECTION] Exporting {weights_path.name} -> TFLite")
    try:
        result = model.export(
            format="tflite",
            imgsz=640,
            dynamic=False,
            simplify=True,
            project=str(export_dir),
            name=run_name,
            device="cpu",
        )
    except Exception as e:
        print(f"  ! Export failed: {e}")
        return []

    if result is None:
        print(f"  ! Exporter did not return a path for {run_name}")
        return []
    
    # result can be a single string path or list depending on version
    if isinstance(result, list):
         tflite_path = Path(result[0]).resolve()
    else:
         tflite_path = Path(result).resolve()

    export_dir.mkdir(parents=True, exist_ok=True)
    target_path = export_dir / f"{weights_path.stem}.tflite"
    
    # YOLO export might create a subdir, we want to move the file up if so or rename it
    if tflite_path != target_path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tflite_path), target_path)
        tflite_path = target_path
    print(f"  ✓ Saved {tflite_path}")
    return [tflite_path]


def load_classifier(weights_path: Path) -> tuple[torch.nn.Module, str]:
    checkpoint = torch.load(weights_path, map_location="cpu")
    model_name = checkpoint.get("model_name", checkpoint.get("backbone", "resnet18"))
    model = create_classifier(model_name=model_name, num_classes=2)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, model_name


def export_classifier_artifacts(
    model: torch.nn.Module,
    base_name: str,
    export_dir: Path,
) -> Dict[str, Optional[Path]]:
    """Export a classifer to TFLite via an intermediate ONNX model."""
    export_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Optional[Path]] = {"tflite": None}
    dummy = torch.randn(1, 3, 224, 224)

    # We need to use a temporary directory for all intermediate artifacts.
    # This prevents us from polluting the export directory.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        tmp_onnx_path = tmpdir_path / f"{base_name}.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                model,
                dummy,
                str(tmp_onnx_path),
                input_names=["images"],
                output_names=["logits"],
                opset_version=13,
                dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
            )

        tflite_path = export_dir / f"{base_name}.tflite"
        print(f"[CLASSIFICATION] Exporting TFLite -> {tflite_path.name}")

        try:
            import tensorflow as tf

            # 1. Convert ONNX to TF SavedModel using onnx2tf CLI
            tf_model_path = tmpdir_path / "tf_model"
            cmd = [
                sys.executable,
                "-m",
                "onnx2tf",
                "-i",
                str(tmp_onnx_path),
                "-o",
                str(tf_model_path),
                "-b",
                "1",  # Set batch size to 1 for compatibility
                "--non_verbose",
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ! onnx2tf conversion failed: {result.stderr}")
                return outputs

            # 2. Convert TF SavedModel to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            outputs["tflite"] = tflite_path
            print(f"  ✓ Saved {tflite_path}")

        except ImportError:
            print("  ! Skipping TFLite export: tensorflow is not installed.")
        except Exception as e:
            print(f"  ! Failed to convert to TFLite: {e}")

    return outputs


def summarize(group: str, conversions: List[Dict[str, Optional[Path]]]) -> None:
    if not conversions:
        print(f"[SUMMARY] No {group} models were exported.")
        return
    print(f"[SUMMARY] Exported {len(conversions)} {group} model(s):")
    for record in conversions:
        for fmt, path in record.items():
            if path is None:
                continue
            print(f"  - {fmt}: {path}")


def run() -> None:
    args = parse_args()
    yolo_dir = args.yolo_dir.resolve() # Changed from detection_dir
    resnet_dir = args.resnet_dir.resolve() # Changed from classification_dir
    export_root = args.export_dir.resolve()
    export_root.mkdir(parents=True, exist_ok=True)

    # Relaxed filtering: Process all .pt and .pth files in the respective dirs
    detection_weights = list_weight_files(yolo_dir, ".pt") # Using yolo_dir
    classification_weights = list_weight_files(resnet_dir, ".pth") # Using resnet_dir

    detection_results: List[Dict[str, Optional[Path]]] = []
    classification_results: List[Dict[str, Optional[Path]]] = []

    detection_export_dir = export_root / "detection"
    classification_export_dir = export_root / "classification"

    if detection_weights:
        detection_export_dir.mkdir(parents=True, exist_ok=True)
        for weight_path in detection_weights:
            if "optimizer" in weight_path.stem: # Skip optimizer states if any
                continue
                
            try:
                written_paths = export_detection_model(weight_path, detection_export_dir)
            except RuntimeError as exc:
                print(f"[DETECTION] Skipping {weight_path.name}: {exc}")
                continue
            detection_results.append(
                {f"tflite_{idx}": path for idx, path in enumerate(written_paths)}
            )
    else:
        print(f"[INFO] No detection checkpoints (*.pt) found in {yolo_dir}") # Using yolo_dir

    if classification_weights:
        classification_export_dir.mkdir(parents=True, exist_ok=True)
        for weight_path in classification_weights:
            print(f"[CLASSIFICATION] Loading {weight_path.name}")
            try:
                model, model_name = load_classifier(weight_path)
            except (ValueError, RuntimeError) as exc:
                print(f"  ! Failed to load {weight_path.name}: {exc}")
                continue

            base_name = f"{weight_path.stem}_{model_name}"
            outputs = export_classifier_artifacts(model, base_name, classification_export_dir)
            classification_results.append(outputs)
    else:
        print(f"[INFO] No classification checkpoints (*.pth) found in {resnet_dir}") # Using resnet_dir

    summarize("detection", detection_results)
    summarize("classification", classification_results)


if __name__ == "__main__":
    run()
