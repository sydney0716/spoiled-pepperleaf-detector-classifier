import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison():
    # Metric definitions
    metrics = ['Recall', 'Precision', 'F1-Score']
    
    # Data
    # Model 1: Small (ResNet18)
    model_small_name = 'ResNet18'
    model_small_scores = [0.942, 0.938, 0.940]

    # Model 2: Large (ResNet50)
    model_large_name = 'ResNet50'
    model_large_scores = [0.951, 0.945, 0.948]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Plotting columns - mimicking the provided style
    # The provided code uses default colors (C0, C1, etc.) implicitly
    rects1 = ax.bar(x - width/2, model_small_scores, width, label=model_small_name)
    rects2 = ax.bar(x + width/2, model_large_scores, width, label=model_large_name)

    ax.set_ylabel('Score')
    ax.set_title('Resnet Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    
    # Y-axis scaling logic from reference
    all_scores = model_small_scores + model_large_scores
    ymax = max(1.0, max(all_scores) * 1.1)
    ax.set_ylim(0, ymax)
    
    ax.legend()

    # Annotation logic from reference
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