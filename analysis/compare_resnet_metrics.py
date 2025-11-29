import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison():
    # Project root setup (assuming script is in analysis/)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RESULTS_DIR = PROJECT_ROOT / "results"

    # Define model result paths
    # Assuming 'resnet18' is the small model and 'resnet50' is the large one.
    # Adjust these paths if your folder structure is different (e.g. results/resnet18/result.json)
    small_model_path = RESULTS_DIR / "resnet18/result.json"
    large_model_path = RESULTS_DIR / "resnet50/result.json"

    def get_metrics(json_path):
        """Reads result.json and calculates Precision, Recall, F1 for the TEST set."""
        if not json_path.exists():
            print(f"[WARN] Results file not found: {json_path}")
            return [0.0, 0.0, 0.0]

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Confusion Matrix is [ [TN, FP], [FN, TP] ] for binary classification
        # "confusion_matrix_test": [[TN, FP], [FN, TP]]
        cm = data.get("confusion_matrix_test")
        if not cm:
            return [0.0, 0.0, 0.0]
        
        tn, fp = cm[0]
        fn, tp = cm[1]

        # Calculate Metrics
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return [recall, precision, f1]

    # Extract metrics from the JSON files
    model_small_scores = get_metrics(small_model_path)
    model_large_scores = get_metrics(large_model_path)

    # Metric definitions for the plot
    metrics = ['Recall', 'Precision', 'F1-Score']
    
    model_small_name = 'ResNet18 (Small)'
    model_large_name = 'ResNet50 (Large)'

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Plotting columns
    rects1 = ax.bar(x - width/2, model_small_scores, width, label=model_small_name)
    rects2 = ax.bar(x + width/2, model_large_scores, width, label=model_large_name)

    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison: Small vs Large Architecture')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    
    # Y-axis scaling logic
    all_scores = model_small_scores + model_large_scores
    # Handle case where all scores might be 0 if files are missing
    max_score = max(all_scores) if all_scores else 1.0
    ymax = max(1.0, max_score * 1.1)
    ax.set_ylim(0, ymax)
    
    ax.legend()

    # Annotation logic
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height + ymax * 0.02,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=8
            )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    
    output_path = 'comparison_plot.png'
    print(f"Saving plot to {output_path}...")
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    plot_comparison()
