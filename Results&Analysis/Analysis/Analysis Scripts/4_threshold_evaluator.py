import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import f1_score


def generate_histogram(df, filename, threshold=0.5):
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
    plt.axvline(x=threshold, color='black', linestyle='--', alpha=0.7, linewidth=1.5,
                label=f'Decision Threshold ({threshold})')

    # Clean formatting
    plt.xlabel('Drone Probability (log scale)', fontsize=14)
    plt.ylabel('Count (log scale)', fontsize=14)
    plt.title(f'{filename}\nBins: {num_bins} | Total: {total_samples:,} samples', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Set x-axis ticks to show readable decimal values
    from matplotlib.ticker import FormatStrFormatter
    x_ticks = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]
    valid_ticks = [tick for tick in x_ticks if min_prob <= tick <= max_prob]
    plt.xticks(valid_ticks)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"  Drone class: mean={prob_1.mean():.4f}, std={prob_1.std():.4f}")
    print(f"  No Drone class: mean={prob_0.mean():.4f}, std={prob_0.std():.4f}")


def process_csv_files_in_directory(directory='.', threshold=0.006):
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
            # Generate histogram with the user-specified threshold
            generate_histogram(df, filename, threshold)

            # Compute classification metrics based on threshold
            preds = (df['drone_probability'] >= threshold).astype(int)
            f1 = f1_score(df['true_label'], preds)
            accuracy = (preds == df['true_label']).mean()
            error_rate = 1 - accuracy

            print(f"Threshold: {threshold}")
            print(f"F1 score: {f1:.4f}")
            print(f"Correct predictions: {accuracy * 100:.2f}%")
            print(f"False predictions: {error_rate * 100:.2f}%")
        except Exception as e:
            print(f"  Error processing {filename}: {e}")


if __name__ == "__main__":
    # Ask user for probability threshold (default should be 0.006 if left blank)
    threshold_input = input("Enter probability threshold [default: 0.006]: ")
    try:
        threshold = float(threshold_input) if threshold_input else 0.006
    except ValueError:
        print("Invalid input, using default threshold = 0.006")
        threshold = 0.006

    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    process_csv_files_in_directory(directory, threshold)
