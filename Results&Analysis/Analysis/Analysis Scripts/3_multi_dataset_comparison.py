import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def generate_combined_histogram(directory='.'):
    """Generate a grid of subplots with one histogram per dataset"""
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return

    # Find all detailed CSV files
    csv_files = [f for f in os.listdir(directory)
                 if f.lower().endswith('_detailed_files.csv')]

    if not csv_files:
        print(f"No detailed CSV files found in directory: {directory}")
        return

    print(f"Found {len(csv_files)} detailed CSV file(s) in directory: {directory}")

    dataset_info = []

    # Process each CSV file
    for filename in sorted(csv_files):
        csv_path = os.path.join(directory, filename)
        dataset_name = filename.replace('_detailed_files.csv', '')

        try:
            df = pd.read_csv(csv_path)

            # Extract probabilities by true label
            prob_0 = df[df['true_label'] == 0]['drone_probability']  # No drone samples
            prob_1 = df[df['true_label'] == 1]['drone_probability']  # Drone samples

            # Calculate performance metrics
            tn = (prob_0 < 0.5).sum()
            tp = (prob_1 >= 0.5).sum()
            fn = (prob_1 < 0.5).sum()
            fp = (prob_0 >= 0.5).sum()

            accuracy = (tn + tp) / len(df) if len(df) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            dataset_info.append({
                'name': dataset_name,
                'prob_0': prob_0,
                'prob_1': prob_1,
                'total': len(df),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'drone_mean': prob_1.mean() if len(prob_1) > 0 else 0,
                'no_drone_mean': prob_0.mean() if len(prob_0) > 0 else 0
            })

            print(f"  Loaded {dataset_name}: {len(df)} samples, Accuracy: {accuracy:.3f}")

        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue

    if not dataset_info:
        print("No valid datasets found")
        return

    # Sort datasets by accuracy for better ordering
    dataset_info.sort(key=lambda x: x['accuracy'], reverse=True)

    # Calculate subplot grid dimensions
    n_datasets = len(dataset_info)
    if n_datasets <= 4:
        rows, cols = 2, 2
    elif n_datasets <= 6:
        rows, cols = 2, 3
    elif n_datasets <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 4, 3

    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    fig.suptitle('Binary Classification Performance - All Datasets\n(Ranked by Accuracy)',
                 fontsize=16, fontweight='bold', y=0.98)

    # Flatten axes array for easier indexing
    if n_datasets == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()

    # Calculate global parameters for consistent scaling
    all_data = []
    for info in dataset_info:
        all_data.extend(info['prob_0'].values)
        all_data.extend(info['prob_1'].values)

    all_probs = np.array(all_data)
    epsilon = 1e-6
    global_min = max(all_probs.min(), epsilon)
    global_max = min(all_probs.max(), 1 - epsilon)

    # Create consistent bins for all plots
    num_bins = 40
    bins = np.logspace(np.log10(global_min), np.log10(global_max), num_bins)

    # Plot each dataset in its own subplot
    for i, info in enumerate(dataset_info):
        ax = axes[i]

        # Plot both classes
        if len(info['prob_0']) > 0:
            ax.hist(info['prob_0'], bins=bins, alpha=0.6,
                    label=f'No Drone (n={len(info["prob_0"]):,})',
                    color='red', density=True, histtype='stepfilled')

        if len(info['prob_1']) > 0:
            ax.hist(info['prob_1'], bins=bins, alpha=0.6,
                    label=f'Drone (n={len(info["prob_1"]):,})',
                    color='blue', density=True, histtype='stepfilled')

        # Add threshold line
        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.8, linewidth=2)

        # Set log scale
        ax.set_yscale('log')
        ax.set_xscale('log')

        # Formatting for each subplot
        ax.set_xlabel('Drone Probability', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)

        # Title with performance metrics
        separation = info['drone_mean'] - info['no_drone_mean']
        title = f"{info['name']}\n"
        title += f"Acc: {info['accuracy']:.3f} | F1: {info['f1']:.3f} | Sep: {separation:.3f}"
        ax.set_title(title, fontsize=11, fontweight='bold')

        # Legend
        ax.legend(fontsize=9, loc='upper center')

        # Grid
        ax.grid(True, alpha=0.3)

        # Set consistent x-axis range and ticks
        ax.set_xlim(global_min, global_max)
        x_ticks = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
        valid_ticks = [tick for tick in x_ticks if global_min <= tick <= global_max]
        ax.set_xticks(valid_ticks)

        # Format x-axis labels
        from matplotlib.ticker import FormatStrFormatter
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    # Hide empty subplots
    for i in range(n_datasets, len(axes)):
        axes[i].set_visible(False)

    # Add overall summary text
    total_samples = sum(info['total'] for info in dataset_info)
    avg_accuracy = np.mean([info['accuracy'] for info in dataset_info])
    best_dataset = dataset_info[0]  # Already sorted by accuracy
    worst_dataset = dataset_info[-1]

    summary_text = f"Overall Summary: {n_datasets} datasets, {total_samples:,} total samples\n"
    summary_text += f"Average Accuracy: {avg_accuracy:.3f} | "
    summary_text += f"Best: {best_dataset['name']} ({best_dataset['accuracy']:.3f}) | "
    summary_text += f"Worst: {worst_dataset['name']} ({worst_dataset['accuracy']:.3f})"

    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)  # Make room for title and summary
    plt.show()

    # Print comprehensive summary table
    print("\n" + "="*100)
    print("COMPREHENSIVE DATASET COMPARISON (Ranked by Accuracy)")
    print("="*100)
    print(f"{'Rank':<4} {'Dataset':<12} {'Accuracy':<8} {'F1':<6} {'Precision':<9} {'Recall':<7} {'Drone Mean':<10} {'No-Drone Mean':<12} {'Separation':<10} {'Samples':<8}")
    print("-" * 100)

    for rank, info in enumerate(dataset_info, 1):
        separation = info['drone_mean'] - info['no_drone_mean']
        print(f"{rank:<4} {info['name']:<12} {info['accuracy']:<8.3f} {info['f1']:<6.3f} {info['precision']:<9.3f} "
              f"{info['recall']:<7.3f} {info['drone_mean']:<10.3f} {info['no_drone_mean']:<12.3f} {separation:<10.3f} {info['total']:<8,}")

    print("="*100)
    print("INTERPRETATION GUIDE:")
    print("• Accuracy: Overall classification accuracy (higher = better)")
    print("• F1: Harmonic mean of precision and recall (higher = better)")
    print("• Separation: Drone Mean - No-Drone Mean (larger = better class separation)")
    print("• Good performance: Blue bars (drone) clustered right of 0.5, Red bars (no-drone) clustered left of 0.5")
    print("="*100)



def generate_individual_histogram(df, filename):
    """Original function to generate individual histograms (kept for compatibility)"""
    # Extract probabilities by true label
    prob_0 = df[df['true_label'] == 0]['drone_probability']
    prob_1 = df[df['true_label'] == 1]['drone_probability']
    all_probs = df['drone_probability']

    # High resolution binning - more bins for better fidelity
    total_samples = len(df)
    if total_samples < 1000:
        num_bins = 100
    elif total_samples < 10000:
        num_bins = 200
    elif total_samples < 8000:
        num_bins = 500
    else:
        num_bins = 2000

    # For log scale x-axis, we need to handle values near 0 and 1
    # Add small epsilon to avoid log(0) issues
    epsilon = 1e-10
    min_prob = max(all_probs.min(), epsilon)
    max_prob = min(all_probs.max(), 1 - epsilon)

    # Create log-spaced bins for x-axis
    bins = np.logspace(np.log10(min_prob), np.log10(max_prob), num_bins)

    # Create clean plot
    plt.figure(figsize=(12, 8))

    # Plot histograms with clean styling
    plt.hist(prob_0, bins=bins, alpha=0.6, label=f'No Drone (n={len(prob_0):,})',
             color='red', density=False)
    plt.hist(prob_1, bins=bins, alpha=0.6, label=f'Drone (n={len(prob_1):,})',
             color='blue', density=False)

    # Use log scale for both axes
    plt.yscale('log')
    plt.xscale('log')

    # Add threshold line
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, linewidth=1.5, label='Decision Threshold')

    # Clean formatting
    plt.xlabel('Drone Probability (log scale)', fontsize=14)
    plt.ylabel('Count (log scale)', fontsize=14)
    plt.title(f'{filename}\nBins: {num_bins} | Total: {total_samples:,} samples', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Set x-axis ticks to show readable decimal values
    x_ticks = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]
    # Only use ticks that are within our data range
    valid_ticks = [tick for tick in x_ticks if min_prob <= tick <= max_prob]
    plt.xticks(valid_ticks)

    # Format x-axis labels to show decimal values
    from matplotlib.ticker import FormatStrFormatter
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"  Drone class: mean={prob_1.mean():.4f}, std={prob_1.std():.4f}")
    print(f"  No Drone class: mean={prob_0.mean():.4f}, std={prob_0.std():.4f}")


def process_csv_files_individually(directory='.'):
    """Process CSV files individually (original behavior)"""
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return

    csv_files = [f for f in os.listdir(directory) if f.lower().endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in directory: {directory}")
        return

    print(f"Found {len(csv_files)} CSV file(s) in directory: {directory}")

    for filename in csv_files:
        csv_path = os.path.join(directory, filename)
        print(f"\nProcessing {filename}...")
        try:
            df = pd.read_csv(csv_path)
            generate_individual_histogram(df, filename)
        except Exception as e:
            print(f"  Error processing {filename}: {e}")


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'

    # Check if user wants individual plots
    if len(sys.argv) > 2 and sys.argv[2] == '--individual':
        print("Generating individual plots for each dataset...")
        process_csv_files_individually(directory)
    else:
        print("Generating combined plot for all datasets...")
        generate_combined_histogram(directory)
