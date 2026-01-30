import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import det_curve, roc_auc_score
from scipy.stats import norm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np


def compute_eer(fpr, fnr):
    """Compute Equal Error Rate - where FPR equals FNR"""
    idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2


def plot_combined_det_curves_with_zoom():
    # Get all CSV files in the current directory
    csv_files = glob.glob("*.csv")

    if not csv_files:
        print("No CSV files found in the current directory.")
        return

    # Define colors and line styles for different augmentations
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    line_styles = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--']
    line_widths = [2.5, 2.5, 3.0, 2.5, 2.5, 2.5, 3.0, 2.5, 2.5, 2.5]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create zoomed inset (40% x 40% size)
    ax_inset = inset_axes(ax, width="40%", height="40%", loc='lower left',
                          bbox_to_anchor=(0.12, 0.12, 1, 1), bbox_transform=ax.transAxes)

    # Store metrics for comparison table
    metrics = []

    for idx, csv_file in enumerate(sorted(csv_files)):
        print(f"Processing: {csv_file}")

        # Read CSV file
        df = pd.read_csv(csv_file)

        # Check for required columns
        if 'true_label' in df.columns and 'drone_probability' in df.columns:
            y_true = df['true_label'].values
            y_scores = df['drone_probability'].values
        else:
            print(f"  ERROR: Required columns not found in {csv_file}")
            continue

        # Compute DET curve
        fpr, fnr, thresholds = det_curve(y_true, y_scores)

        # Compute metrics
        auc = roc_auc_score(y_true, y_scores)
        eer = compute_eer(fpr, fnr)

        # Extract label from filename for legend
        label = csv_file.replace('_Fusion_detailed_files_1Epoch.csv', '').replace('_', ' ')

        # Store metrics
        metrics.append({
            'Augmentation': label,
            'AUC': auc,
            'EER': eer * 100  # as percentage
        })

        # Plot on probit scale - main plot
        color = colors[idx % len(colors)]
        linestyle = line_styles[idx % len(line_styles)]
        linewidth = line_widths[idx % len(line_widths)]

        # Main plot
        ax.plot(norm.ppf(fpr), norm.ppf(fnr), label=f"{label} (EER: {eer*100:.2f}%)",
                color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.8)

        # Inset plot (zoomed)
        ax_inset.plot(norm.ppf(fpr), norm.ppf(fnr), color=color,
                      linestyle=linestyle, linewidth=linewidth, alpha=0.8)

    # Set up probit scale axes - main plot
    ticks = [0.001, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    tick_labels = ['0.1%', '1.0%', '5.0%', '10.0%', '20.0%', '30.0%', '40.0%', '50.0%']
    tick_locations = norm.ppf(ticks)

    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_locations)
    ax.set_yticklabels(tick_labels)

    ax.set_xlim(norm.ppf(0.001), norm.ppf(0.5))
    ax.set_ylim(norm.ppf(0.001), norm.ppf(0.5))

    ax.set_xlabel('False Positive Rate (probit scale)', fontsize=12)
    ax.set_ylabel('False Negative Rate (probit scale)', fontsize=12)
    ax.set_title('DET Curve Comparison - Augmentation Methods', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Set up zoomed inset - focus on the area where curves might differ
    # Zoom to approximately 9.0-10.5% FPR and FNR range (extreme zoom)
    zoom_ticks = [0.2895, 0.2905, 0.290]
    zoom_tick_labels = ['28.9%', '29.0%', '29.1%']
    zoom_tick_locations = norm.ppf(zoom_ticks)

    ax_inset.set_xlim(norm.ppf(0.2895), norm.ppf(0.2905))
    ax_inset.set_ylim(norm.ppf(0.2895), norm.ppf(0.2905))
    ax_inset.set_xticks(zoom_tick_locations)
    ax_inset.set_xticklabels(zoom_tick_labels, fontsize=8)
    ax_inset.set_yticks(zoom_tick_locations)
    ax_inset.set_yticklabels(zoom_tick_labels, fontsize=8)
    ax_inset.grid(True, linestyle='--', alpha=0.5)
    ax_inset.set_title('Zoomed View (Extreme)', fontsize=9)

    # Draw box around zoomed region on main plot
    mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--')

    plt.tight_layout()
    plt.savefig('combined_det_curves_zoomed.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nPlot saved as 'combined_det_curves_zoomed.png'")

    # Print metrics comparison table
    print("\n" + "="*60)
    print("PERFORMANCE METRICS COMPARISON")
    print("="*60)
    print(f"{'Augmentation':<20} {'AUC':>10} {'EER (%)':>12}")
    print("-"*60)
    for m in metrics:
        print(f"{m['Augmentation']:<20} {m['AUC']:>10.4f} {m['EER']:>12.2f}")
    print("="*60)

    # Compute and display pairwise differences
    if len(metrics) > 1:
        print("\nPAIRWISE EER DIFFERENCES (percentage points):")
        print("-"*60)
        for i in range(len(metrics)):
            for j in range(i+1, len(metrics)):
                diff = abs(metrics[i]['EER'] - metrics[j]['EER'])
                print(f"{metrics[i]['Augmentation']} vs {metrics[j]['Augmentation']}: {diff:.3f} pp")

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('augmentation_comparison_metrics.csv', index=False)
    print("\nMetrics saved to 'augmentation_comparison_metrics.csv'")


if __name__ == "__main__":
    plot_combined_det_curves_with_zoom()

