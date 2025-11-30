from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "runs" / "detection"

YOLO_RUNS: Dict[str, Path] = {
    "YOLOv8n": RESULTS_DIR / "yolo8n" / "results.csv",
    "YOLOv8s": RESULTS_DIR / "yolo8s" / "results.csv",
    "YOLOv11n": RESULTS_DIR / "yolo11n" / "results.csv",
    "YOLOv11s": RESULTS_DIR / "yolo11s" / "results.csv",
}

DINO_METRICS = RESULTS_DIR / "grounding_dino" / "val_metrics.csv"

OUTPUT_FIG = PROJECT_ROOT / "yolo_comparison_plot.png"

def load_best_row(csv_path: Path) -> Optional[pd.Series]:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    # YOLOv8/v11 CSV headers often have spaces; strip them
    df.columns = [c.strip() for c in df.columns]
    
    if "metrics/mAP50-95(B)" not in df.columns:
        # Fallback or check if it's DINO format
        if "AP" in df.columns: # DINO
            return df.iloc[0]
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

    # Wrap single path in list for iteration
    for path in [DINO_METRICS]:
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
    
    return rows

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
    draw_chart(rows)


if __name__ == "__main__":
    main()
