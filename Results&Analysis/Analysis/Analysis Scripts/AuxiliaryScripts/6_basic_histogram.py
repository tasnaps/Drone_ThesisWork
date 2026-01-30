import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def generate_histogram(df, csv_filepath):
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

    # Extract just the filename for the title
    filename = os.path.basename(csv_filepath)
    plt.title(f'{filename}\nBins: {num_bins} | Total: {total_samples:,} samples', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Set x-axis ticks to show readable decimal values
    from matplotlib.ticker import LogFormatterSciNotation, FixedLocator
    x_ticks = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]
    # Only use ticks that are within our data range
    valid_ticks = [tick for tick in x_ticks if min_prob <= tick <= max_prob]
    plt.xticks(valid_ticks)

    # Format x-axis labels to show decimal values
    from matplotlib.ticker import FormatStrFormatter
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    plt.tight_layout()

    # Generate output filename by replacing .csv with .png
    base_name = os.path.splitext(csv_filepath)[0]  # Remove .csv extension
    plot_filepath = f"{base_name}_histogram.png"

    # Save the plot instead of showing it
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    print(f"  Plot saved to: {plot_filepath}")

    # Print summary statistics
    print(f"  Drone class: mean={prob_1.mean():.4f}, std={prob_1.std():.4f}")
    print(f"  No Drone class: mean={prob_0.mean():.4f}, std={prob_0.std():.4f}")


def find_csv_files_recursively(directory='.'):
    """
    Recursively find all CSV files in the directory and subdirectories.
    Uses os.walk() to ensure deep traversal through all subfolders.
    """
    csv_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.csv'):
                full_path = os.path.join(root, file)
                csv_files.append(full_path)

    return csv_files


def check_csv_columns(csv_filepath):
    """
    Check if the CSV file has the required columns:
    file_id, true_label, predicted_label, drone_probability, aggregation_method, aggregation_threshold, split
    """
    required_columns = {
        'file_id', 'true_label', 'predicted_label', 'drone_probability',
        'aggregation_method', 'aggregation_threshold', 'split'
    }

    try:
        # Read just the header to check columns
        df_header = pd.read_csv(csv_filepath, nrows=0)
        available_columns = set(df_header.columns)

        # Check if all required columns are present
        return required_columns.issubset(available_columns)

    except Exception as e:
        print(f"  Error reading CSV header from {csv_filepath}: {e}")
        return False


def process_csv_files(directory='.'):
    """
    Find and process all CSV files recursively in the directory.
    Only process files that have the required columns.
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return

    csv_files = find_csv_files_recursively(directory)

    if not csv_files:
        print(f"No CSV files found in directory: {directory}")
        return

    print(f"Found {len(csv_files)} CSV file(s) in directory tree: {directory}")
    print("Checking files for required columns...")

    processed_count = 0
    skipped_count = 0

    for csv_filepath in csv_files:
        print(f"\nChecking {csv_filepath}...")

        # Check if file has required columns
        if check_csv_columns(csv_filepath):
            print(f"  ✓ File has required columns - processing...")
            try:
                df = pd.read_csv(csv_filepath)
                generate_histogram(df, csv_filepath)
                processed_count += 1

            except Exception as e:
                print(f"  Error processing {csv_filepath}: {e}")
                skipped_count += 1
        else:
            print(f"  ✗ File does not have required columns - skipping...")
            skipped_count += 1

    print(f"\n=== Summary ===")
    print(f"Total CSV files found: {len(csv_files)}")
    print(f"Files processed: {processed_count}")
    print(f"Files skipped: {skipped_count}")


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    process_csv_files(directory)
