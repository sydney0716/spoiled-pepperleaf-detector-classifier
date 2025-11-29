#!/usr/bin/env python3
"""
Summarize detection metrics for YOLO runs and Grounding DINO into a single CSV.

Outputs runs/model_comparison.csv with columns:
model, best_epoch, mAP50-95, mAP50, precision, recall
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS_DIR = Path("runs")
YOLO_RUNS: Dict[str, Path] = {
    "YOLOv8n": RUNS_DIR / "yolov8n" / "results.csv",
    "YOLOv8s": RUNS_DIR / "yolov8s" / "results.csv",
    "YOLOv11n": RUNS_DIR / "yolov11n" / "results.csv",
    "YOLOv11s": RUNS_DIR / "yolov11s" / "results.csv",
}

DINO_METRICS = [
    Path("runs/grounding_dino_val_metrics.csv"),
    Path("detection_yolo/.splits/src_split_v20_s42/annotations/grounding_dino_val_metrics.csv"),
]

OUTPUT_CSV = RUNS_DIR / "model_comparison.csv"
OUTPUT_FIG = RUNS_DIR / "model_comparison.png"


def load_best_row(csv_path: Path) -> Optional[pd.Series]:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    if "metrics/mAP50-95(B)" not in df.columns:
        return df.iloc[-1]
    idx = df["metrics/mAP50-95(B)"].idxmax()
    return df.loc[idx]


def summarize() -> List[Dict[str, Optional[float]]]:
    rows: List[Dict[str, Optional[float]]] = []

    for name, csv_path in YOLO_RUNS.items():
        best = load_best_row(csv_path)
        if best is None:
            continue
        rows.append(
            {
                "model": name,
                "best_epoch": int(best["epoch"]),
                "mAP50-95": float(best["metrics/mAP50-95(B)"]),
                "mAP50": float(best["metrics/mAP50(B)"]),
                "precision": float(best["metrics/precision(B)"]),
                "recall": float(best["metrics/recall(B)"]),
            }
        )

    for path in DINO_METRICS:
        best = load_best_row(path)
        if best is None:
            continue
        rows.append(
            {
                "model": best.get("model", "GroundingDINO"),
                "best_epoch": None,
                "mAP50-95": float(best.get("AP", float("nan"))),
                "mAP50": float(best.get("AP50", float("nan"))),
                "precision": None,
                "recall": float(best.get("AR100", float("nan"))),
            }
        )
        break

    return rows


def write_csv(rows: List[Dict[str, Optional[float]]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "best_epoch", "mAP50-95", "mAP50", "precision", "recall"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[OK] Wrote summary to {OUTPUT_CSV}")


def draw_chart(rows: List[Dict[str, Optional[float]]]) -> None:
    df = pd.DataFrame(rows)
    df = df.set_index("model")
    metrics = ["mAP50-95", "mAP50"]
    if any(m not in df.columns for m in metrics):
        return
    values = df[metrics].astype(float).fillna(0.0)
    x = np.arange(len(values))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, values[metrics[0]], width, label=metrics[0])
    ax.bar(x + width / 2, values[metrics[1]], width, label=metrics[1])

    ax.set_ylabel("Score")
    ax.set_title("Detection Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(values.index, rotation=15, ha="right")
    ymax = max(1.0, values.to_numpy().max() * 1.1)
    ax.set_ylim(0, ymax)
    ax.legend()

    for idx, model in enumerate(values.index):
        for offset, metric in zip((-width / 2, width / 2), metrics):
            score = values.loc[model, metric]
            ax.text(
                idx + offset,
                score + ymax * 0.02,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_FIG, dpi=200)
    plt.close(fig)
    print(f"[OK] Wrote chart to {OUTPUT_FIG}")


def main() -> None:
    rows = summarize()
    if not rows:
        raise SystemExit("No metrics found; ensure YOLO runs and Grounding DINO metrics exist.")
    write_csv(rows)
    draw_chart(rows)


if __name__ == "__main__":
    main()
