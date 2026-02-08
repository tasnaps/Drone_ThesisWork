import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from collections import defaultdict


def extract_model_dataset_and_epoch_info(csv_filepath):
    """
    Extract model name, dataset name and epoch number from the file path.

    Handles multiple folder structures:
    1. .../CNN_LSTM/100Epoch/DatasetName/Dataset_detailed_files.csv
    2. .../CNN_LSTM/100Epoch/Dataset_detailed_files.csv (CSV directly in epoch folder)
    3. .../Wav2Vec2_AllAugments/Epoch1/datasets/Dataset_detailed_files.csv
    4. .../ModelName/EpochX/subfolder/Dataset_detailed_files.csv

    Epoch folder patterns: "100Epoch", "10Epoch", "Epoch1", "Epoch10", etc.
    """
    import re

    path_parts = csv_filepath.split(os.sep)
    csv_filename = path_parts[-1]

    # Extract dataset name from CSV filename (pattern: DatasetName_detailed_files.csv)
    dataset_name = None
    filename_match = re.match(r'^(.+?)_detailed_files\.csv$', csv_filename, re.IGNORECASE)
    if filename_match:
        dataset_name = filename_match.group(1)

    # Find epoch folder - handles both "100Epoch" and "Epoch100" patterns
    epoch_folder = None
    epoch_index = None
    epoch_num = None

    for i, part in enumerate(path_parts):
        # Match patterns like "100Epoch", "10Epoch", "Epoch1", "Epoch10", etc.
        epoch_match = re.match(r'^(\d+)epoch$|^epoch(\d+)$', part, re.IGNORECASE)
        if epoch_match:
            epoch_folder = part
            epoch_index = i
            # Extract the number from whichever group matched
            epoch_num = int(epoch_match.group(1) or epoch_match.group(2))
            break

    # Find model name - it should be the folder before the epoch folder
    model_name = None
    if epoch_index is not None and epoch_index > 0:
        potential_model = path_parts[epoch_index - 1]
        # Skip if it looks like a results/root folder
        if not re.match(r'^results|^output|^data$', potential_model, re.IGNORECASE):
            model_name = potential_model

    # If dataset_name not found from filename, try to find from folder structure
    if dataset_name is None and epoch_index is not None:
        # Check parent folder of CSV (but skip 'datasets' folder)
        parent_folder = path_parts[-2] if len(path_parts) >= 2 else None

        if parent_folder:
            # Skip utility folders
            skip_folders = {'datasets', 'plots', 'reports', 'summaries', 'archive', 'models', 'runs'}
            if parent_folder.lower() not in skip_folders and 'epoch' not in parent_folder.lower():
                # Check if it's not a timestamp folder
                if not re.search(r'evaluation_\d{8}_\d{6}', parent_folder):
                    dataset_name = parent_folder

    return model_name, dataset_name, epoch_num



def check_csv_columns(csv_filepath):
    """
    Check if the CSV file has the required columns.
    """
    required_columns = {
        'file_id', 'true_label', 'predicted_label', 'drone_probability',
        'aggregation_method', 'aggregation_threshold', 'split'
    }

    try:
        df_header = pd.read_csv(csv_filepath, nrows=0)
        available_columns = set(df_header.columns)
        return required_columns.issubset(available_columns)
    except Exception as e:
        print(f"  Error reading CSV header from {csv_filepath}: {e}")
        return False


def find_csv_files_recursively(directory='.'):
    """
    Recursively find all CSV files in the directory and subdirectories.
    """
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.csv'):
                full_path = os.path.join(root, file)
                csv_files.append(full_path)
    return csv_files


def calculate_weighted_stats(probabilities, num_segments=20):
    """
    Calculate statistics with density weighting for visualization.
    Returns segments with their densities for weighted line plotting.
    """
    if len(probabilities) == 0:
        return [], [], []
    
    # Create bins for density calculation
    bins = np.linspace(0, 1, num_segments + 1)
    counts, _ = np.histogram(probabilities, bins=bins)
    
    # Calculate segment centers and densities
    segment_centers = (bins[:-1] + bins[1:]) / 2
    segment_densities = counts / len(probabilities)  # Normalize densities
    
    # Filter out empty segments
    non_zero_indices = counts > 0
    segment_centers = segment_centers[non_zero_indices]
    segment_densities = segment_densities[non_zero_indices]
    
    return segment_centers, segment_densities, bins


def find_calibration_threshold(csv_filepath):
    """
    Find the calibrated threshold from CalibrationDataset_detailed_files.csv in the same epoch folder.
    This is used as a fallback when the current CSV has threshold of 0.0 (due to rounding issues).

    Searches in:
    1. Same directory as the CSV file
    2. Parent directory (for cases where CSVs are in 'datasets' subfolder)
    3. Sibling directories (e.g., if CSV is in datasets/, look in parent for CalibrationDataset folder)
    """
    import glob

    csv_dir = os.path.dirname(csv_filepath)
    parent_dir = os.path.dirname(csv_dir)

    # List of possible locations to search for calibration file
    search_paths = [
        # Same directory
        os.path.join(csv_dir, 'CalibrationDataset_detailed_files.csv'),
        os.path.join(csv_dir, 'Calibration*_detailed_files.csv'),
        # Parent directory
        os.path.join(parent_dir, 'CalibrationDataset_detailed_files.csv'),
        os.path.join(parent_dir, 'Calibration*_detailed_files.csv'),
        # Sibling CalibrationDataset folder
        os.path.join(parent_dir, 'CalibrationDataset', 'CalibrationDataset_detailed_files.csv'),
        os.path.join(parent_dir, 'CalibrationDataset', '*_detailed_files.csv'),
        # datasets subfolder pattern
        os.path.join(parent_dir, 'datasets', 'CalibrationDataset_detailed_files.csv'),
        os.path.join(parent_dir, 'datasets', 'Calibration*_detailed_files.csv'),
    ]

    for pattern in search_paths:
        matches = glob.glob(pattern)
        for calibration_file in matches:
            if os.path.exists(calibration_file) and calibration_file != csv_filepath:
                try:
                    df_cal = pd.read_csv(calibration_file, nrows=10)  # Only read a few rows
                    if 'aggregation_threshold' in df_cal.columns:
                        threshold_values = df_cal['aggregation_threshold'].dropna()
                        if len(threshold_values) > 0:
                            threshold = float(threshold_values.iloc[0])
                            if threshold > 0.0:  # Valid threshold found
                                return threshold, calibration_file
                except Exception:
                    continue

    return None, None


def calculate_dataset_stats(df, csv_filepath=None):
    """
    Calculate min, median, max statistics for drone probabilities in a dataset.
    Separates statistics by true_label: 1 = drone, 0 = unknown/no drone
    Also calculates density information for weighted visualization.
    Extracts the calibrated aggregation_threshold from the data.
    If threshold is 0.0, attempts to find it from CalibrationDataset_detailed_files.csv.
    """
    all_probs = df['drone_probability']

    # Separate by true label
    drone_probs = df[df['true_label'] == 1]['drone_probability']
    unknown_probs = df[df['true_label'] == 0]['drone_probability']

    # Extract aggregation_threshold (calibrated threshold) - should be same for all rows in a file
    aggregation_threshold = 0.5  # default fallback
    threshold_source = 'default'

    if 'aggregation_threshold' in df.columns and len(df) > 0:
        # Get the first non-null value
        threshold_values = df['aggregation_threshold'].dropna()
        if len(threshold_values) > 0:
            aggregation_threshold = float(threshold_values.iloc[0])
            threshold_source = 'csv'

    # If threshold is 0.0 (likely a rounding issue), try to find it from calibration file
    if aggregation_threshold == 0.0 and csv_filepath is not None:
        cal_threshold, cal_file = find_calibration_threshold(csv_filepath)
        if cal_threshold is not None:
            aggregation_threshold = cal_threshold
            threshold_source = f'calibration_file ({os.path.basename(cal_file)})'
            print(f"    ⚠ Threshold was 0.0, retrieved {aggregation_threshold:.10f} from {os.path.basename(cal_file)}")
        else:
            print(f"    ⚠ Threshold is 0.0 and no calibration file found - using 0.5 fallback")
            aggregation_threshold = 0.5
            threshold_source = 'fallback (no calibration file)'

    stats = {
        'overall': {
            'min': all_probs.min(),
            'median': all_probs.median(),
            'max': all_probs.max(),
            'count': len(all_probs),
            'probabilities': all_probs.values
        },
        'threshold': aggregation_threshold,
        'threshold_source': threshold_source
    }

    # Add drone (true_label = 1) statistics if data exists
    if len(drone_probs) > 0:
        drone_centers, drone_densities, _ = calculate_weighted_stats(drone_probs)
        stats['drone'] = {
            'min': drone_probs.min(),
            'median': drone_probs.median(),
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
            'median': unknown_probs.median(),
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


def generate_dataset_epoch_plot(epoch_data, epoch_num, output_dir, model_name='Unknown', threshold=0.5):
    """
    Generate a plot showing dataset statistics for a specific epoch with weighted density visualization.
    Shows separate statistics for drone and unknown sounds.
    Y-axis: datasets, X-axis: probability statistics (min, median, max)
    Line thickness varies based on data concentration in different probability ranges.
    Uses the calibrated aggregation_threshold instead of hardcoded 0.5.
    """
    if not epoch_data:
        return

    # Prepare data for plotting
    datasets = list(epoch_data.keys())
    datasets.sort()  # Sort for consistent ordering

    # Use accessible colors (colorblind-friendly palette)
    accessible_colors = [
        '#0077BB',  # Blue
        '#EE7733',  # Orange
        '#009988',  # Teal
        '#CC3311',  # Red
        '#33BBEE',  # Cyan
        '#EE3377',  # Magenta
        '#BBBBBB',  # Grey
        '#000000',  # Black
        '#44AA99',  # Teal green
        '#882255',  # Wine
        '#DDCC77',  # Sand
        '#117733',  # Green
        '#88CCEE',  # Light blue
        '#AA4499',  # Purple
        '#999933',  # Olive
        '#661100',  # Dark red
        '#6699CC',  # Steel blue
        '#AA4466',  # Rose
        '#332288',  # Indigo
        '#DDDDDD',  # Light grey
    ]
    colors = [accessible_colors[i % len(accessible_colors)] for i in range(len(datasets))]

    # Create subplot with more height to accommodate both categories
    fig, ax = plt.subplots(figsize=(16, max(12, int(len(datasets) * 1.8))))

    # Create positions for datasets - each dataset gets 2 rows (drone + unknown)
    # Increased spacing to prevent label overlap
    y_spacing = 1.2
    y_positions = []
    y_labels = []

    for i, dataset in enumerate(datasets):
        base_y = i * 2 * y_spacing
        y_positions.extend([base_y + 0.3, base_y + 1.2])  # drone, unknown - more separation
        y_labels.extend([f'{dataset} (Drone)', f'{dataset} (Unknown)'])

    # Add alternating background shading for each dataset group
    for i, dataset in enumerate(datasets):
        base_y = i * 2 * y_spacing
        # Alternate between light grey and white backgrounds
        if i % 2 == 0:
            ax.axhspan(base_y - 0.1, base_y + y_spacing * 2 - 0.3,
                      facecolor='#F5F5F5', alpha=0.7, zorder=0)

        # Add separator line between dataset groups (except before first)
        if i > 0:
            separator_y = base_y - 0.2
            ax.axhline(y=separator_y, color='#CCCCCC', linestyle='-', linewidth=1.5, zorder=1)

    for i, dataset in enumerate(datasets):
        stats = epoch_data[dataset]
        color = colors[i]

        # Calculate positions for this dataset
        drone_y = i * 2 * y_spacing + 0.3
        unknown_y = i * 2 * y_spacing + 1.2

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
                verticalalignment='center', fontsize=18, color='darkgreen')
        ax.text(1.02, unknown_y, f'n={unknown_count:,}',
                transform=ax.get_yaxis_transform(),
                verticalalignment='center', fontsize=18, color='darkred')

    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=18)
    ax.set_xlabel('Drone Probability', fontsize=22)
    ax.set_ylabel('Dataset & Category', fontsize=22)
    ax.set_title(f'{model_name} - Dataset Statistics - Epoch {epoch_num}\n(○ = Drone Median, □ = Unknown Median, Line Thickness = Data Concentration)\nCalibrated Threshold: {threshold:.4f}',
                fontsize=22, pad=20)
    ax.tick_params(axis='x', labelsize=16)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='x')

    # Add threshold line using calibrated threshold
    ax.axvline(x=threshold, color='red', linestyle='-', alpha=0.8, linewidth=3,
               label=f'Calibrated Threshold ({threshold:.4f})')

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=6, alpha=0.8, label='Drone (thick = high concentration)'),
        Line2D([0], [0], color='black', linewidth=6, alpha=0.6, linestyle='--', label='Unknown (thick = high concentration)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=14, label='Median (Drone)', linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=14, label='Median (Unknown)', linestyle='None'),
        Line2D([0], [0], color='red', linewidth=3, label=f'Calibrated Threshold ({threshold:.4f})')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08),
              ncol=3, fontsize=16, frameon=True)

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
    Process CSV files and organize data by model, epoch and dataset.
    Generate plots showing dataset statistics across epochs with weighted visualization.

    Handles folder structures:
    1. .../CNN_LSTM/100Epoch/DatasetName/Dataset_detailed_files.csv
    2. .../CNN_LSTM/100Epoch/Dataset_detailed_files.csv
    3. .../Wav2Vec2_AllAugments/Epoch1/datasets/Dataset_detailed_files.csv
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return

    # Create output directory on desktop
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    output_dir = os.path.join(desktop_path, 'dataset_epoch_analysis_weighted')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    csv_files = find_csv_files_recursively(directory)

    if not csv_files:
        print(f"No CSV files found in directory: {directory}")
        return

    print(f"Found {len(csv_files)} CSV file(s) in directory tree: {directory}")

    # Organize data by model, epoch and dataset
    # Structure: model -> epoch -> dataset -> stats
    model_epoch_dataset_data = defaultdict(lambda: defaultdict(dict))
    processed_count = 0
    skipped_count = 0

    for csv_filepath in csv_files:
        print(f"\nProcessing {csv_filepath}...")

        # Check if file has required columns
        if not check_csv_columns(csv_filepath):
            print(f"  ✗ File does not have required columns - skipping...")
            skipped_count += 1
            continue

        # Extract model, dataset and epoch info using the updated function
        model_name, dataset_name, epoch_num = extract_model_dataset_and_epoch_info(csv_filepath)

        if dataset_name is None or epoch_num is None:
            print(f"  ✗ Could not extract dataset/epoch info from path - skipping...")
            skipped_count += 1
            continue

        # Use 'Unknown_Model' if model couldn't be determined
        if model_name is None:
            model_name = 'Unknown_Model'

        print(f"  ✓ Model: {model_name}, Dataset: {dataset_name}, Epoch: {epoch_num}")

        try:
            df = pd.read_csv(csv_filepath)
            stats = calculate_dataset_stats(df, csv_filepath)

            # Store stats by model, epoch, dataset
            model_epoch_dataset_data[model_name][epoch_num][dataset_name] = stats
            processed_count += 1

        except Exception as e:
            print(f"  Error processing {csv_filepath}: {e}")
            skipped_count += 1

    # Generate plots for each model
    print(f"\n=== Generating Weighted Plots ===")

    for model_name in sorted(model_epoch_dataset_data.keys()):
        model_data = model_epoch_dataset_data[model_name]

        # Create model-specific output directory
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        print(f"\n--- Processing Model: {model_name} ---")

        epochs = sorted(model_data.keys())

        for epoch_num in epochs:
            print(f"\nGenerating weighted plot for {model_name} - Epoch {epoch_num}...")

            # Extract calibrated threshold from any dataset in this epoch (should be same for all)
            epoch_datasets = model_data[epoch_num]
            calibrated_threshold = 0.5  # fallback default
            for dataset_stats in epoch_datasets.values():
                if 'threshold' in dataset_stats:
                    calibrated_threshold = dataset_stats['threshold']
                    break

            print(f"  Using calibrated threshold: {calibrated_threshold:.4f}")
            generate_dataset_epoch_plot(model_data[epoch_num], epoch_num, model_output_dir, model_name, calibrated_threshold)

        # Generate summary across all epochs for this model
        generate_summary_across_epochs(model_data, model_output_dir, model_name)

    print(f"\n=== Summary ===")
    print(f"Total CSV files found: {len(csv_files)}")
    print(f"Files processed: {processed_count}")
    print(f"Files skipped: {skipped_count}")
    print(f"Models processed: {len(model_epoch_dataset_data)}")
    for model_name, model_data in model_epoch_dataset_data.items():
        print(f"  {model_name}: {len(model_data)} epochs")
    print(f"Output directory: {output_dir}")


def generate_summary_across_epochs(epoch_dataset_data, output_dir, model_name='Unknown'):
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

    # Extract calibrated thresholds per epoch
    epoch_thresholds = {}
    for epoch in epochs:
        # Get threshold from any dataset in this epoch
        for dataset_stats in epoch_dataset_data[epoch].values():
            if 'threshold' in dataset_stats:
                epoch_thresholds[epoch] = dataset_stats['threshold']
                break
        if epoch not in epoch_thresholds:
            epoch_thresholds[epoch] = 0.5  # fallback

    # Generate accessible colors for datasets (colorblind-friendly palette)
    accessible_colors = [
        '#0077BB',  # Blue
        '#EE7733',  # Orange
        '#009988',  # Teal
        '#CC3311',  # Red
        '#33BBEE',  # Cyan
        '#EE3377',  # Magenta
        '#BBBBBB',  # Grey
        '#000000',  # Black
        '#44AA99',  # Teal green
        '#882255',  # Wine
        '#DDCC77',  # Sand
        '#117733',  # Green
        '#88CCEE',  # Light blue
        '#AA4499',  # Purple
        '#999933',  # Olive
        '#661100',  # Dark red
        '#6699CC',  # Steel blue
        '#AA4466',  # Rose
        '#332288',  # Indigo
        '#DDDDDD',  # Light grey
    ]
    colors = [accessible_colors[i % len(accessible_colors)] for i in range(len(all_datasets))]

    # Create two separate figures: one for drone, one for unknown

    # === DRONE STATISTICS ===
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 18))

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
                    color=colors[i], linewidth=2, markersize=8)

    ax1.set_xlabel('Epoch', fontsize=18)
    ax1.set_ylabel('Median Drone Probability', fontsize=18)
    ax1.set_title(f'{model_name} - Median Drone Probability Across Epochs (Drone Samples Only)', fontsize=20)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.grid(True, alpha=0.3)
    # Plot calibrated threshold line (varies by epoch)
    threshold_epochs = sorted(epoch_thresholds.keys())
    threshold_values = [epoch_thresholds[e] for e in threshold_epochs]
    ax1.plot(threshold_epochs, threshold_values, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Calibrated Threshold', marker='x')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=14, frameon=True)

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
                    color=colors[i], linewidth=2, markersize=8)

    ax2.set_xlabel('Epoch', fontsize=18)
    ax2.set_ylabel('Min Drone Probability', fontsize=18)
    ax2.set_title(f'{model_name} - Minimum Drone Probability Across Epochs (Drone Samples Only)', fontsize=20)
    ax2.tick_params(axis='both', labelsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.plot(threshold_epochs, threshold_values, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Calibrated Threshold', marker='x')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=14, frameon=True)

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
                    color=colors[i], linewidth=2, markersize=8)

    ax3.set_xlabel('Epoch', fontsize=22)
    ax3.set_ylabel('Max Drone Probability', fontsize=22)
    ax3.set_title(f'{model_name} - Maximum Drone Probability Across Epochs (Drone Samples Only)', fontsize=24)
    ax3.tick_params(axis='both', labelsize=18)
    ax3.grid(True, alpha=0.3)
    ax3.plot(threshold_epochs, threshold_values, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Calibrated Threshold', marker='x')
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=16, frameon=True)

    plt.tight_layout(h_pad=3.0)

    # Save drone summary plot
    drone_summary_filepath = os.path.join(output_dir, 'summary_across_epochs_DRONE_weighted.png')
    plt.savefig(drone_summary_filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved drone summary plot: {drone_summary_filepath}")

    # === UNKNOWN STATISTICS ===
    fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 18))

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
                    color=colors[i], linewidth=2, markersize=8)

    ax1.set_xlabel('Epoch', fontsize=18)
    ax1.set_ylabel('Median Drone Probability', fontsize=18)
    ax1.set_title(f'{model_name} - Median Drone Probability Across Epochs (Unknown Samples Only)', fontsize=20)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.plot(threshold_epochs, threshold_values, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Calibrated Threshold', marker='x')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=14, frameon=True)

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
                    color=colors[i], linewidth=2, markersize=8)

    ax2.set_xlabel('Epoch', fontsize=18)
    ax2.set_ylabel('Min Drone Probability', fontsize=18)
    ax2.set_title(f'{model_name} - Minimum Drone Probability Across Epochs (Unknown Samples Only)', fontsize=20)
    ax2.tick_params(axis='both', labelsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.plot(threshold_epochs, threshold_values, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Calibrated Threshold', marker='x')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=14, frameon=True)

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
                    color=colors[i], linewidth=2, markersize=8)

    ax3.set_xlabel('Epoch', fontsize=22)
    ax3.set_ylabel('Max Drone Probability', fontsize=22)
    ax3.set_title(f'{model_name} - Maximum Drone Probability Across Epochs (Unknown Samples Only)', fontsize=24)
    ax3.tick_params(axis='both', labelsize=18)
    ax3.grid(True, alpha=0.3)
    ax3.plot(threshold_epochs, threshold_values, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Calibrated Threshold', marker='x')
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=16, frameon=True)

    plt.tight_layout(h_pad=3.0)

    # Save unknown summary plot
    unknown_summary_filepath = os.path.join(output_dir, 'summary_across_epochs_UNKNOWN_weighted.png')
    plt.savefig(unknown_summary_filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved unknown summary plot: {unknown_summary_filepath}")


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    process_datasets_by_epoch(directory)
