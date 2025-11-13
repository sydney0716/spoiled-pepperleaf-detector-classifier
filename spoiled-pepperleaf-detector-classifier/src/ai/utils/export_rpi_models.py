#!/usr/bin/env python3
"""
Convert trained pepper leaf models into Raspberry Pi ready artifacts.

The script searches the training output folders for YOLO detection weights
(`*.pt`) and ResNet classification checkpoints (`*.pth`) and exports them into
smaller files that can be executed on a Raspberry Pi Compute Module 5.

Detection models are exported to float32 and float16 TFLite blobs via the
Ultralytics exporter. Classification checkpoints are converted to TorchScript
and ONNX by default; if TensorFlow + onnx-tf are available a float16 TFLite
version of the classifier is produced as well. All outputs land in
`weights/exported` (override with `--export-dir`).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torchvision import models


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATHS: Dict[str, Path] = {
    "detection": REPO_ROOT / "runs" / "detection",
    "classification": REPO_ROOT / "weights" / "trained",
    "export": REPO_ROOT / "weights" / "exported",
}

CLASSIFIER_ARCH = {
    "resnet18": {"builder": models.resnet18, "hidden_features": 256},
    "resnet50": {"builder": models.resnet50, "hidden_features": 512},
}


def load_ultralytics() -> Any:
    try:
        from ultralytics import YOLO as _YOLO  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Ultralytics is not installed. Run `pip install ultralytics` to enable detection exports."
        ) from exc
    return _YOLO


def create_classifier(backbone: str, num_classes: int = 2) -> nn.Module:
    if backbone not in CLASSIFIER_ARCH:
        raise ValueError(f"Unsupported backbone '{backbone}'. Expected one of {tuple(CLASSIFIER_ARCH)}.")
    config = CLASSIFIER_ARCH[backbone]
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
        "--detection-dir",
        type=Path,
        default=DEFAULT_PATHS["detection"],
        help="Directory that stores YOLO detection checkpoints (*.pt).",
    )
    parser.add_argument(
        "--classification-dir",
        type=Path,
        default=DEFAULT_PATHS["classification"],
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
    """Export a YOLO checkpoint to float32/float16 TFLite."""
    YOLOCls = load_ultralytics()
    model = YOLOCls(str(weights_path))
    written: List[Path] = []
    for tag, half in (("fp32", False), ("fp16", True)):
        run_name = f"{weights_path.stem}_{tag}"
        print(f"[DETECTION] Exporting {weights_path.name} -> {tag} TFLite")
        try:
            result = model.export(
                format="tflite",
                imgsz=640,
                half=half,
                int8=False,
                dynamic=False,
                simplify=True,
                project=str(export_dir),
                name=run_name,
                device="cpu",
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Ultralytics requires TensorFlow for TFLite export. "
                "Install `tensorflow` (CPU-only is sufficient) and retry."
            ) from exc

        if result is None:
            print(f"  ! Exporter did not return a path for {run_name}")
            continue
        tflite_path = Path(result)
        written.append(tflite_path)
        print(f"  ✓ Saved {tflite_path}")
    return written


def load_classifier(weights_path: Path) -> tuple[torch.nn.Module, str]:
    checkpoint = torch.load(weights_path, map_location="cpu")
    backbone = checkpoint.get("backbone", "resnet18")
    model = create_classifier(backbone=backbone, num_classes=2)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, backbone


def export_classifier_artifacts(
    model: torch.nn.Module,
    base_name: str,
    export_dir: Path,
) -> Dict[str, Optional[Path]]:
    export_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Optional[Path]] = {}
    dummy = torch.randn(1, 3, 224, 224)

    torchscript_path = export_dir / f"{base_name}_torchscript.pt"
    print(f"[CLASSIFICATION] Exporting TorchScript -> {torchscript_path.name}")
    with torch.inference_mode():
        traced = torch.jit.trace(model, dummy)
        traced.save(torchscript_path)
    outputs["torchscript"] = torchscript_path

    onnx_path = export_dir / f"{base_name}.onnx"
    print(f"[CLASSIFICATION] Exporting ONNX -> {onnx_path.name}")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["images"],
            output_names=["logits"],
            opset_version=13,
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        )
    outputs["onnx"] = onnx_path

    tflite_path = maybe_export_classifier_tflite(onnx_path, export_dir, base_name)
    outputs["tflite"] = tflite_path
    return outputs


def maybe_export_classifier_tflite(onnx_path: Path, export_dir: Path, base_name: str) -> Optional[Path]:
    """Convert ONNX -> TF SavedModel -> float16 TFLite if the deps are installed."""
    try:
        import onnx  # type: ignore
        from onnx_tf.backend import prepare  # type: ignore
        import tensorflow as tf  # type: ignore
    except ModuleNotFoundError:
        print(
            "  ! Skipping TFLite export for classifier (install `onnx`, `onnx-tf`, and "
            "`tensorflow` to enable)."
        )
        return None

    print("  - Converting ONNX -> TensorFlow SavedModel")
    onnx_model = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model)

    tmp_saved_model = export_dir / f"{base_name}_tf"
    if tmp_saved_model.exists():
        shutil.rmtree(tmp_saved_model)
    tf_rep.export_graph(str(tmp_saved_model))

    tflite_path = export_dir / f"{base_name}_fp16.tflite"
    print(f"  - Converting SavedModel -> {tflite_path.name}")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(tmp_saved_model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)
    shutil.rmtree(tmp_saved_model, ignore_errors=True)
    print(f"  ✓ Saved {tflite_path}")
    return tflite_path


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
    detection_dir = args.detection_dir.resolve()
    classification_dir = args.classification_dir.resolve()
    export_root = args.export_dir.resolve()
    export_root.mkdir(parents=True, exist_ok=True)

    detection_weights = list_weight_files(detection_dir, ".pt")
    classification_weights = list_weight_files(classification_dir, ".pth")

    detection_results: List[Dict[str, Optional[Path]]] = []
    classification_results: List[Dict[str, Optional[Path]]] = []

    detection_export_dir = export_root / "detection"
    classification_export_dir = export_root / "classification"

    if detection_weights:
        detection_export_dir.mkdir(parents=True, exist_ok=True)
        for weight_path in detection_weights:
            try:
                written_paths = export_detection_model(weight_path, detection_export_dir)
            except RuntimeError as exc:
                print(f"[DETECTION] Skipping {weight_path.name}: {exc}")
                continue
            detection_results.append(
                {f"tflite_{idx}": path for idx, path in enumerate(written_paths)}
            )
    else:
        print(f"[INFO] No detection checkpoints (*.pt) found in {detection_dir}")

    if classification_weights:
        classification_export_dir.mkdir(parents=True, exist_ok=True)
        for weight_path in classification_weights:
            print(f"[CLASSIFICATION] Loading {weight_path.name}")
            try:
                model, backbone = load_classifier(weight_path)
            except (ValueError, RuntimeError) as exc:
                print(f"  ! Failed to load {weight_path.name}: {exc}")
                continue
            base_name = f"{weight_path.stem}_{backbone}"
            outputs = export_classifier_artifacts(model, base_name, classification_export_dir)
            classification_results.append(outputs)
    else:
        print(f"[INFO] No classification checkpoints (*.pth) found in {classification_dir}")

    summarize("detection", detection_results)
    summarize("classification", classification_results)


if __name__ == "__main__":
    run()
