import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def find_csv_files_recursively(directory='.'):
    """
    Recursively find all CSV files that seem to contain evaluation results.
    """
    csv_files = []
    # A pattern to identify relevant CSVs, e.g., ending in '_detailed_files.csv'
    file_pattern = re.compile(r'_detailed_files\.csv$', re.IGNORECASE)

    for root, _, files in os.walk(directory):
        for file in files:
            if file_pattern.search(file):
                full_path = os.path.join(root, file)
                csv_files.append(full_path)

    return csv_files


def check_csv_columns(csv_filepath):
    """
    Check if the CSV file contains the required columns: 'true_label' and 'drone_probability'.
    """
    try:
        df_head = pd.read_csv(csv_filepath, nrows=0)
        required_cols = {'true_label', 'drone_probability'}
        return required_cols.issubset(df_head.columns)
    except Exception:
        return False


def extract_dataset_and_epoch_info(csv_filepath):
    """
    Robustly extract (dataset_name, epoch_num) from a CSV file path.
    - Epoch is taken from the nearest ancestor directory matching: 'Epoch5', 'Epoch-5', 'epoch_5', or '5Epoch'.
    - Dataset is the nearest non-generic, non-evaluation/timestamp ancestor not identified as an epoch folder.
    - Fallback to filename by stripping common suffixes like '_detailed_files', '_results', etc.
    Returns:
        (dataset_name: Optional[str], epoch_num: Optional[int])
    """
    import os
    import re
    from pathlib import Path

    p = Path(csv_filepath)

    # --- Epoch extraction ---
    epoch_num = None
    # Named groups to avoid group-index ambiguity
    epoch_dir_re = re.compile(
        r'^(?:epoch|ep)[-_]?(?P<num1>\d+)$|^(?P<num2>\d+)[-_]?epoch$',
        re.IGNORECASE
    )

    def _dir_is_epoch(name: str):
        m = epoch_dir_re.match(name)
        if m:
            g = m.group('num1') or m.group('num2')
            return int(g)
        return None

    # Prefer closest ancestor directory that looks like an epoch
    for ancestor in p.parents:  # parent, grandparent, ...
        val = _dir_is_epoch(ancestor.name)
        if val is not None:
            epoch_num = val
            break

    # Fallback: search anywhere in the path string (e.g., uncommon layouts)
    if epoch_num is None:
        m = re.search(
            r'(?:epoch|ep)[-_]?(\d+)|(\d+)[-_]?epoch', str(p), re.IGNORECASE
        )
        if m:
            g = m.group(1) or m.group(2)
            if g:
                epoch_num = int(g)

    # --- Dataset extraction ---
    dataset_name = None

    GENERIC_NAMES = {
        'datasets', 'dataset', 'data', 'csv', 'files', 'file',
        'results', 'result', 'output', 'outputs', 'reports',
        'graphs', 'plots', 'images', 'figures', 'stats', 'statistics'
    }

    def _looks_like_eval_or_model(name: str) -> bool:
        # Matches 'evaluation_20250101_123456', 'eval-20250101', 'model_20250101_123456', etc.
        if re.search(r'(evaluation|eval|model)[-_]?\d{6,}', name, re.IGNORECASE):
            return True
        if re.search(r'\d{8}[-_]\d{6}', name):  # 20250101_123456
            return True
        return False

    # Choose nearest non-generic, non-eval/timestamp, non-epoch ancestor as dataset
    for ancestor in p.parents:
        name = ancestor.name
        if _dir_is_epoch(name) is not None:
            continue
        lname = name.lower()
        if lname in GENERIC_NAMES:
            continue
        if _looks_like_eval_or_model(name):
            continue
        dataset_name = name
        break

    # Fallback to filename-based heuristic if no ancestor qualified
    if not dataset_name:
        stem = p.stem  # filename without extension
        # Strip common suffixes used by exports
        COMMON_SUFFIXES = [
            '_detailed_files', '_detailed', '_files', '_results',
            '_predictions', '_probs', '_scores', '-detailed-files',
            '-detailed', '-files', '-results', '-predictions'
        ]
        s_lower = stem.lower()
        changed = True
        while changed:
            changed = False
            for suf in COMMON_SUFFIXES:
                if s_lower.endswith(suf):
                    stem = stem[: -len(suf)]
                    s_lower = stem.lower()
                    changed = True
                    break

        # If still too generic, try taking the first meaningful token
        token = re.split(r'[_\-\.\s]+', stem)[0] if stem else ''
        token_lower = token.lower()

        if token and token_lower not in GENERIC_NAMES and not _looks_like_eval_or_model(token):
            dataset_name = token
        else:
            # Keep None to let caller decide to skip if dataset is unknown
            dataset_name = None

    return dataset_name, epoch_num


def calculate_weighted_stats(probabilities, num_bins=100):
    """
    Calculate weighted statistics using Kernel Density Estimation (KDE).
    Returns centers of density bins and their corresponding weights (density).
    """
    if len(probabilities) < 2:  # KDE requires at least 2 points
        return np.array([]), np.array([]), None

    # Create a KDE
    kde = gaussian_kde(probabilities)

    # Define the range for evaluation
    p_min, p_max = probabilities.min(), probabilities.max()
    x_grid = np.linspace(p_min, p_max, num_bins)

    # Evaluate the KDE on the grid
    density = kde.evaluate(x_grid)

    return x_grid, density, kde


def calculate_dataset_stats(df):
    """
    Calculate statistics for a given dataset DataFrame.
    Separates data by true_label (1 for drone, 0 for unknown).
    """
    stats = {}

    # Extract aggregation threshold. Assume it's the same for the whole file.
    # Default to 0.5 if not present.
    if not df.empty and 'aggregation_threshold' in df.columns and not df['aggregation_threshold'].empty:
        stats['aggregation_threshold'] = df['aggregation_threshold'].iloc[0]
    else:
        stats['aggregation_threshold'] = 0.5

    # Separate probabilities by true_label
    drone_probs = df[df['true_label'] == 1]['drone_probability']
    unknown_probs = df[df['true_label'] == 0]['drone_probability']

    # Add drone (true_label = 1) statistics if data exists
    if len(drone_probs) > 0:
        drone_centers, drone_densities, _ = calculate_weighted_stats(drone_probs)
        stats['drone'] = {
            'min': drone_probs.min(),
            'median': np.median(drone_probs),
            'max': drone_probs.max(),
            'count': len(drone_probs),
            'probabilities': drone_probs.values,
            'density_centers': drone_centers,
            'density_weights': drone_densities
        }
    else:
        stats['drone'] = {
            'min': 0.0,
            'median': 0.0,
            'max': 0.0,
            'count': 0,
            'probabilities': np.array([]),
            'density_centers': np.array([]),
            'density_weights': np.array([])
        }

    # Add unknown (true_label = 0) statistics if data exists
    if len(unknown_probs) > 0:
        unknown_centers, unknown_densities, _ = calculate_weighted_stats(unknown_probs)
        stats['unknown'] = {
            'min': unknown_probs.min(),
            'median': np.median(unknown_probs),
            'max': unknown_probs.max(),
            'count': len(unknown_probs),
            'probabilities': unknown_probs.values,
            'density_centers': unknown_centers,
            'density_weights': unknown_densities
        }
    else:
        stats['unknown'] = {
            'min': 0.0,
            'median': 0.0,
            'max': 0.0,
            'count': 0,
            'probabilities': np.array([]),
            'density_centers': np.array([]),
            'density_weights': np.array([])
        }

    return stats


def plot_weighted_density_line(ax, y_pos, stats_category, base_color, line_style='-', alpha_base=0.8):
    """
    Plot a weighted density line where thickness/opacity varies based on data concentration.
    """
    if stats_category['count'] == 0:
        return

    density_centers = stats_category['density_centers']
    density_weights = stats_category['density_weights']

    if len(density_centers) == 0:
        return

    # Normalize weights for better visualization (scale to reasonable line widths)
    max_weight = np.max(density_weights) if len(density_weights) > 0 else 1
    min_linewidth = 1
    max_linewidth = 8

    # Plot individual segments with varying thickness
    for i in range(len(density_centers) - 1):
        x_start = density_centers[i]
        x_end = density_centers[i + 1]
        weight = density_weights[i]

        # Calculate line width and alpha based on density
        normalized_weight = weight / max_weight if max_weight > 0 else 0
        line_width = min_linewidth + (max_linewidth - min_linewidth) * normalized_weight
        alpha = alpha_base * (0.3 + 0.7 * normalized_weight)  # Vary alpha too

        ax.plot([x_start, x_end], [y_pos, y_pos],
                color=base_color, linewidth=line_width, alpha=alpha,
                linestyle=line_style, solid_capstyle='round')

    # Add a thin continuous line for the full range for context
    ax.plot([stats_category['min'], stats_category['max']], [y_pos, y_pos],
            color=base_color, linewidth=0.5, alpha=0.3, linestyle=line_style)


def generate_dataset_epoch_plot(epoch_data, epoch_num, output_dir):
    """
    Generate a plot showing dataset statistics for a specific epoch with weighted density visualization.
    Shows separate statistics for drone and unknown sounds.
    Y-axis: datasets, X-axis: probability statistics (min, median, max)
    Line thickness varies based on data concentration in different probability ranges.
    """
    if not epoch_data:
        return

    # Prepare data for plotting
    datasets = list(epoch_data.keys())
    datasets.sort()  # Sort for consistent ordering

    # Get the aggregation threshold. Assume it's the same for all datasets in the epoch.
    # Default to 0.5 if not found.
    first_dataset_key = datasets[0] if datasets else None
    aggregation_threshold = epoch_data.get(first_dataset_key, {}).get('aggregation_threshold', 0.5)

    # Generate unique colors for each dataset
    colors = plt.cm.tab20(np.linspace(0, 1, len(datasets)))

    # Create subplot with more height to accommodate both categories
    fig, ax = plt.subplots(figsize=(16, max(10, int(len(datasets) * 1.2))))

    # Create positions for datasets - each dataset gets 2 rows (drone + unknown)
    y_spacing = 1.0
    y_positions = []
    y_labels = []

    for i, dataset in enumerate(datasets):
        base_y = i * 2 * y_spacing
        y_positions.extend([base_y + 0.2, base_y + 0.8])  # drone, unknown
        y_labels.extend([f'{dataset}\n(Drone)', f'{dataset}\n(Unknown)'])

    for i, dataset in enumerate(datasets):
        stats = epoch_data[dataset]
        color = colors[i]

        # Calculate positions for this dataset
        drone_y = i * 2 * y_spacing + 0.2
        unknown_y = i * 2 * y_spacing + 0.8

        # Plot drone statistics with weighted density (true_label = 1)
        if stats['drone']['count'] > 0:
            # Plot weighted density line
            plot_weighted_density_line(ax, drone_y, stats['drone'], color, '-', 0.8)

            # Plot median as a point
            ax.scatter(stats['drone']['median'], drone_y, color=color, s=120, zorder=5,
                       marker='o', edgecolor='black', linewidth=2)

        # Plot unknown statistics with weighted density (true_label = 0)
        if stats['unknown']['count'] > 0:
            # Plot weighted density line with different style
            plot_weighted_density_line(ax, unknown_y, stats['unknown'], color, '--', 0.6)

            # Plot median as a point with different marker
            ax.scatter(stats['unknown']['median'], unknown_y, color=color, s=120, zorder=5,
                       marker='s', edgecolor='black', linewidth=2)

        # Add sample counts as text
        drone_count = stats['drone']['count']
        unknown_count = stats['unknown']['count']

        ax.text(1.02, drone_y, f'n={drone_count:,}',
                transform=ax.get_yaxis_transform(),
                verticalalignment='center', fontsize=9, color='darkgreen')
        ax.text(1.02, unknown_y, f'n={unknown_count:,}',
                transform=ax.get_yaxis_transform(),
                verticalalignment='center', fontsize=9, color='darkred')

    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel('Drone Probability', fontsize=14)
    ax.set_ylabel('Dataset & Category', fontsize=14)
    ax.set_title(
        f'Dataset Statistics - Epoch {epoch_num}\n(○ = Drone Median, □ = Unknown Median, Line Thickness = Data Concentration)',
        fontsize=16, pad=20)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='x')

    # Add threshold line
    ax.axvline(x=aggregation_threshold, color='red', linestyle='-', alpha=0.8, linewidth=3,
               label=f'Decision Threshold ({aggregation_threshold:.3f})')

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=6, alpha=0.8, label='Drone (thick = high concentration)'),
        Line2D([0], [0], color='black', linewidth=6, alpha=0.6, linestyle='--',
               label='Unknown (thick = high concentration)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Median (Drone)',
               linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10, label='Median (Unknown)',
               linestyle='None'),
        Line2D([0], [0], color='red', linewidth=3, label=f'Decision Threshold ({aggregation_threshold:.3f})')
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save plot
    plot_filename = f'epoch_{epoch_num:02d}_dataset_stats_weighted.png'
    plot_filepath = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved weighted plot: {plot_filepath}")

    # Print summary statistics
    print(f"  Epoch {epoch_num} Summary:")
    for dataset in datasets:
        stats = epoch_data[dataset]
        print(f"    {dataset}:")
        print(f"      Drone: min={stats['drone']['min']:.4f}, median={stats['drone']['median']:.4f}, "
              f"max={stats['drone']['max']:.4f}, samples={stats['drone']['count']:,}")
        print(f"      Unknown: min={stats['unknown']['min']:.4f}, median={stats['unknown']['median']:.4f}, "
              f"max={stats['unknown']['max']:.4f}, samples={stats['unknown']['count']:,}")


def process_datasets_by_epoch(directory='.'):
    """
    Process CSV files and organize data by epoch and dataset.
    Generate plots showing dataset statistics across epochs with weighted visualization.
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return

    # Create output directory on desktop with a timestamp to make it unique
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    output_dir_name = f'dataset_epoch_analysis_weighted_{timestamp}'
    output_dir = os.path.join(desktop_path, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    csv_files = find_csv_files_recursively(directory)

    if not csv_files:
        print(f"No CSV files found in directory: {directory}")
        return

    print(f"Found {len(csv_files)} CSV file(s) in directory tree: {directory}")

    # Organize data by epoch and dataset
    epoch_dataset_data = defaultdict(dict)  # epoch -> dataset -> stats
    processed_count = 0
    skipped_count = 0

    for csv_filepath in csv_files:
        print(f"\nProcessing {csv_filepath}...")

        # Check if file has required columns
        if not check_csv_columns(csv_filepath):
            print(f"  ✗ File does not have required columns - skipping...")
            skipped_count += 1
            continue

        # Extract dataset and epoch info
        dataset_name, epoch_num = extract_dataset_and_epoch_info(csv_filepath)

        if dataset_name is None or epoch_num is None:
            print(f"  ✗ Could not extract dataset/epoch info from path - skipping...")
            skipped_count += 1
            continue

        print(f"  ✓ Dataset: {dataset_name}, Epoch: {epoch_num}")

        try:
            df = pd.read_csv(csv_filepath)
            stats = calculate_dataset_stats(df)

            # Store stats
            epoch_dataset_data[epoch_num][dataset_name] = stats
            processed_count += 1

        except Exception as e:
            print(f"  Error processing {csv_filepath}: {e}")
            skipped_count += 1

    # Generate plots for each epoch
    print(f"\n=== Generating Weighted Plots ===")
    epochs = sorted(epoch_dataset_data.keys())

    for epoch_num in epochs:
        print(f"\nGenerating weighted plot for Epoch {epoch_num}...")
        generate_dataset_epoch_plot(epoch_dataset_data[epoch_num], epoch_num, output_dir)

    # Generate summary across all epochs
    generate_summary_across_epochs(epoch_dataset_data, output_dir)

    print(f"\n=== Summary ===")
    print(f"Total CSV files found: {len(csv_files)}")
    print(f"Files processed: {processed_count}")
    print(f"Files skipped: {skipped_count}")
    print(f"Epochs processed: {len(epochs)}")
    print(f"Output directory: {output_dir}")


def generate_summary_across_epochs(epoch_dataset_data, output_dir):
    """
    Generate summary plots showing how dataset statistics change across epochs.
    Creates separate plots for drone and unknown sound categories.
    """
    if not epoch_dataset_data:
        return

    epochs = sorted(epoch_dataset_data.keys())
    all_datasets = set()

    # Collect all unique datasets
    for epoch_data in epoch_dataset_data.values():
        all_datasets.update(epoch_data.keys())

    all_datasets = sorted(list(all_datasets))

    if not all_datasets:
        return

    # Generate colors for datasets
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_datasets)))

    # Create two separate figures: one for drone, one for unknown

    # === DRONE STATISTICS ===
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16))

    # Plot drone median probabilities across epochs
    for i, dataset in enumerate(all_datasets):
        medians = []
        epoch_list = []

        for epoch in epochs:
            if dataset in epoch_dataset_data[epoch] and epoch_dataset_data[epoch][dataset]['drone']['count'] > 0:
                medians.append(epoch_dataset_data[epoch][dataset]['drone']['median'])
                epoch_list.append(epoch)

        if medians:
            ax1.plot(epoch_list, medians, marker='o', label=dataset,
                     color=colors[i], linewidth=2, markersize=6)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Median Drone Probability', fontsize=12)
    ax1.set_title('Median Drone Probability Across Epochs by Dataset (Drone Samples Only)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold')

    # Plot drone min probabilities across epochs
    for i, dataset in enumerate(all_datasets):
        mins = []
        epoch_list = []

        for epoch in epochs:
            if dataset in epoch_dataset_data[epoch] and epoch_dataset_data[epoch][dataset]['drone']['count'] > 0:
                mins.append(epoch_dataset_data[epoch][dataset]['drone']['min'])
                epoch_list.append(epoch)

        if mins:
            ax2.plot(epoch_list, mins, marker='s', label=dataset,
                     color=colors[i], linewidth=2, markersize=6)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Min Drone Probability', fontsize=12)
    ax2.set_title('Minimum Drone Probability Across Epochs by Dataset (Drone Samples Only)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold')

    # Plot drone max probabilities across epochs
    for i, dataset in enumerate(all_datasets):
        maxs = []
        epoch_list = []

        for epoch in epochs:
            if dataset in epoch_dataset_data[epoch] and epoch_dataset_data[epoch][dataset]['drone']['count'] > 0:
                maxs.append(epoch_dataset_data[epoch][dataset]['drone']['max'])
                epoch_list.append(epoch)

        if maxs:
            ax3.plot(epoch_list, maxs, marker='^', label=dataset,
                     color=colors[i], linewidth=2, markersize=6)

    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Max Drone Probability', fontsize=12)
    ax3.set_title('Maximum Drone Probability Across Epochs by Dataset (Drone Samples Only)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold')

    plt.tight_layout()

    # Save drone summary plot
    drone_summary_filepath = os.path.join(output_dir, 'summary_across_epochs_DRONE_weighted.png')
    plt.savefig(drone_summary_filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved drone summary plot: {drone_summary_filepath}")

    # === UNKNOWN STATISTICS ===
    fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16))

    # Plot unknown median probabilities across epochs
    for i, dataset in enumerate(all_datasets):
        medians = []
        epoch_list = []

        for epoch in epochs:
            if dataset in epoch_dataset_data[epoch] and epoch_dataset_data[epoch][dataset]['unknown']['count'] > 0:
                medians.append(epoch_dataset_data[epoch][dataset]['unknown']['median'])
                epoch_list.append(epoch)

        if medians:
            ax1.plot(epoch_list, medians, marker='o', label=dataset,
                     color=colors[i], linewidth=2, markersize=6)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Median Drone Probability', fontsize=12)
    ax1.set_title('Median Drone Probability Across Epochs by Dataset (Unknown Samples Only)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold')

    # Plot unknown min probabilities across epochs
    for i, dataset in enumerate(all_datasets):
        mins = []
        epoch_list = []

        for epoch in epochs:
            if dataset in epoch_dataset_data[epoch] and epoch_dataset_data[epoch][dataset]['unknown']['count'] > 0:
                mins.append(epoch_dataset_data[epoch][dataset]['unknown']['min'])
                epoch_list.append(epoch)

        if mins:
            ax2.plot(epoch_list, mins, marker='s', label=dataset,
                     color=colors[i], linewidth=2, markersize=6)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Min Drone Probability', fontsize=12)
    ax2.set_title('Minimum Drone Probability Across Epochs by Dataset (Unknown Samples Only)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold')

    # Plot unknown max probabilities across epochs
    for i, dataset in enumerate(all_datasets):
        maxs = []
        epoch_list = []

        for epoch in epochs:
            if dataset in epoch_dataset_data[epoch] and epoch_dataset_data[epoch][dataset]['unknown']['count'] > 0:
                maxs.append(epoch_dataset_data[epoch][dataset]['unknown']['max'])
                epoch_list.append(epoch)

        if maxs:
            ax3.plot(epoch_list, maxs, marker='^', label=dataset,
                     color=colors[i], linewidth=2, markersize=6)

    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Max Drone Probability', fontsize=12)
    ax3.set_title('Maximum Drone Probability Across Epochs by Dataset (Unknown Samples Only)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold')

    plt.tight_layout()

    # Save unknown summary plot
    unknown_summary_filepath = os.path.join(output_dir, 'summary_across_epochs_UNKNOWN_weighted.png')
    plt.savefig(unknown_summary_filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved unknown summary plot: {unknown_summary_filepath}")


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    process_datasets_by_epoch(directory)
