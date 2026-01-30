import pandas as pd
import numpy as np
import os
import sys
import re
from collections import defaultdict

# Pretty-print thresholds with more precision to avoid many ties like 0.0001
THRESHOLD_DECIMALS = 8


def _fmt_threshold(x: float | None) -> str:
    if x is None:
        return 'N/A'
    return f"{float(x):.{THRESHOLD_DECIMALS}f}"


def _fmt_metric(x: float | None, decimals: int = 4) -> str:
    if x is None:
        return 'N/A'
    return f"{float(x):.{decimals}f}"


def extract_model_and_epoch_info(csv_filepath):
    """
    Extract model name and epoch number from the file path.
    Returns: (model_name, epoch_number, dataset_name)
    """
    # Normalize path separators
    path_parts = csv_filepath.replace('\\', '/').split('/')

    # Extract epoch number - look for patterns like 'Epoch-5', '5Epoch', 'Epoch5', '1Epoch', '5-Epoch', etc.
    epoch_num = None
    epoch_folder = None
    for part in path_parts:
        # Match various epoch patterns
        epoch_match = re.search(r'(\d+)[-_]?Epoch|Epoch[-_]?(\d+)', part, re.IGNORECASE)
        if epoch_match:
            epoch_str = epoch_match.group(1) or epoch_match.group(2)
            epoch_num = int(epoch_str)
            epoch_folder = part
            break

    # Extract model name - typically a parent directory above the epoch folder
    model_name = "Unknown"
    if epoch_folder:
        try:
            epoch_idx = path_parts.index(epoch_folder)
            if epoch_idx > 0:
                # Look backwards for a non-generic folder name
                for i in range(epoch_idx - 1, -1, -1):
                    candidate = path_parts[i]
                    if candidate.lower() not in ['datasets', 'results', 'csv', 'data', 'evaluation']:
                        # Avoid using drive letters or very short names
                        if len(candidate) > 2 and ':' not in candidate:
                            model_name = candidate
                            break
        except ValueError:
            pass

    # Extract dataset name from filename
    filename = os.path.basename(csv_filepath)
    stem = os.path.splitext(filename)[0]

    # Remove common suffixes
    if stem.endswith('_detailed_files'):
        dataset_name = stem[:-len('_detailed_files')]
    else:
        dataset_name = stem

    # Rename "Tapio" to "Authors"
    if dataset_name.lower() == 'tapio':
        dataset_name = 'Authors'

    # Handle Wonjun_Yi case
    if 'wonjun' in dataset_name.lower():
        dataset_name = 'Wonjun_Yi'

    return model_name, epoch_num, dataset_name


def calculate_brier_score(df):
    """
    Calculate Brier Score: mean((predicted_probability - true_label)²)
    Lower is better (0 = perfect, 1 = worst)
    Works for both single-class and two-class datasets.
    """
    predicted_probs = df['drone_probability']
    true_labels = df['true_label']

    brier_score = np.mean((predicted_probs - true_labels) ** 2)
    return brier_score


def calculate_metrics(df, threshold):
    """
    Calculate precision, recall, F1, accuracy, and Brier Score using the given threshold.
    For single-class datasets, returns accuracy and Brier Score only (F1 set to None).
    Returns a dict including the threshold used and a flag for single-class.
    """
    # Calculate Brier Score (works for all datasets)
    brier_score = calculate_brier_score(df)

    # Apply threshold to get predictions
    predictions = (df['drone_probability'] >= threshold).astype(int)
    true_labels = df['true_label']

    # Check if we have both classes
    unique_labels = true_labels.unique()
    is_single_class = len(unique_labels) < 2

    # Calculate confusion matrix elements
    tp = ((predictions == 1) & (true_labels == 1)).sum()
    tn = ((predictions == 0) & (true_labels == 0)).sum()
    fp = ((predictions == 1) & (true_labels == 0)).sum()
    fn = ((predictions == 0) & (true_labels == 1)).sum()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0

    if is_single_class:
        # For single-class datasets, only return accuracy and Brier Score
        return {
            'accuracy': accuracy,
            'precision': None,
            'recall': None,
            'f1': None,
            'brier_score': brier_score,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'total_samples': len(df),
            'threshold': threshold,
            'is_single_class': True
        }

    # For two-class datasets, calculate all metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'brier_score': brier_score,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'total_samples': len(df),
        'threshold': threshold,
        'is_single_class': False
    }


def find_optimal_threshold(df, metric='f1', calibrated_threshold=0.5):
    """
    Find the optimal threshold for a dataset.

    Uses a fast bounded optimizer (scipy.optimize.minimize_scalar) and then refines
    on the discrete set of probability cut-points (since F1 changes only when the
    threshold crosses a probability value).

    Args:
        df: DataFrame with drone_probability and true_label columns
        metric: 'f1' (maximize; two-class only) or 'accuracy' (maximize; single-class)
        calibrated_threshold: The current calibrated threshold to use as fallback

    Returns:
        tuple: (optimal_threshold, best_metric_value, metrics_dict)
    """
    from sklearn.metrics import f1_score
    from scipy.optimize import minimize_scalar

    probs = df['drone_probability'].to_numpy(dtype=float)
    y_true = df['true_label'].to_numpy(dtype=int)

    # Single-class check
    unique = np.unique(y_true)
    is_single_class = len(unique) < 2
    if metric == 'f1' and is_single_class:
        return None, None, None

    # Keep away from hard edges. If the data is extremely peaky, this still keeps us sane.
    # We also tighten bounds to the observed probability range where possible.
    eps = 1e-6
    observed_min = float(np.min(probs)) if probs.size else 0.0
    observed_max = float(np.max(probs)) if probs.size else 1.0

    lower = max(0.02, observed_min - eps)
    upper = min(0.98, observed_max + eps)
    if not (lower < upper):
        lower, upper = 0.02, 0.98

    # Define objectives
    if metric == 'f1':
        def obj(t: float) -> float:
            preds = (probs >= t).astype(int)
            return -f1_score(y_true, preds, zero_division=0)
        minimize = True
    elif metric == 'accuracy':
        # Minimize error rate = maximize accuracy
        def obj(t: float) -> float:
            preds = (probs >= t).astype(int)
            return float(np.mean(preds != y_true))
        minimize = True
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # 1) Fast continuous search
    try:
        res = minimize_scalar(obj, bounds=(lower, upper), method='bounded')
        candidate = float(res.x)
    except Exception:
        # SciPy not available / failed -> fallback to calibrated threshold
        candidate = float(calibrated_threshold)

    # 2) Discrete refinement on real cut-points (important for stepwise metrics)
    unique_probs = np.unique(probs)
    unique_probs = unique_probs[(unique_probs > 0) & (unique_probs < 1)]
    if unique_probs.size:
        idx = int(np.searchsorted(unique_probs, candidate))
        neighbor_idxs = {max(0, idx - 2), max(0, idx - 1), min(unique_probs.size - 1, idx), min(unique_probs.size - 1, idx + 1), min(unique_probs.size - 1, idx + 2)}

        candidates = [float(unique_probs[i]) for i in sorted(neighbor_idxs)] + [float(calibrated_threshold)]

        best_t = float(calibrated_threshold)
        best_val = float('inf') if minimize else -float('inf')

        for t in candidates:
            if not (lower <= t <= upper):
                continue
            v = float(obj(t))
            if minimize:
                if v < best_val:
                    best_val = v
                    best_t = t
            else:
                if v > best_val:
                    best_val = v
                    best_t = t

        optimal_threshold = best_t
    else:
        optimal_threshold = float(calibrated_threshold)

    # Compute final metrics dict
    final_metrics = calculate_metrics(df, optimal_threshold)

    if metric == 'f1':
        best_value = final_metrics['f1']
    else:
        best_value = final_metrics['accuracy']

    return optimal_threshold, best_value, final_metrics


def process_csv_for_metrics(csv_filepath):
    """
    Process a single CSV file and extract metrics.
    Returns: (model_name, epoch_num, dataset_name, metrics_dict)
    """
    try:
        df = pd.read_csv(csv_filepath)

        # Get threshold from CSV
        threshold = float(df['aggregation_threshold'].iloc[0])

        # Extract model/epoch/dataset info
        model_name, epoch_num, dataset_name = extract_model_and_epoch_info(csv_filepath)

        # Calculate metrics
        metrics = calculate_metrics(df, threshold)

        if metrics['is_single_class']:
            print(f"  ⚠ Model: {model_name}, Epoch: {epoch_num}, Dataset: {dataset_name}, Single-class (Brier: {metrics['brier_score']:.4f})")
        else:
            print(f"  ✓ Model: {model_name}, Epoch: {epoch_num}, Dataset: {dataset_name}, Brier: {metrics['brier_score']:.4f}, F1: {metrics['f1']:.4f}")

        return model_name, epoch_num, dataset_name, metrics

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None, None, None, None


def process_csv_for_optimized_metrics(csv_filepath):
    """
    Process CSV with both calibrated and dataset-optimized thresholds.

    Returns:
        tuple: (model_name, epoch_num, dataset_name, calibrated_metrics, optimized_metrics, threshold_diff)
    """
    try:
        df = pd.read_csv(csv_filepath)

        # Get calibrated threshold from CSV
        calibrated_threshold = float(df['aggregation_threshold'].iloc[0])

        # Extract model/epoch/dataset info
        model_name, epoch_num, dataset_name = extract_model_and_epoch_info(csv_filepath)

        # Calculate metrics with calibrated threshold
        calibrated_metrics = calculate_metrics(df, calibrated_threshold)

        # Find dataset-optimized threshold (maximize F1, or minimize Brier for single-class)
        if calibrated_metrics['is_single_class']:
            # For single-class, optimize accuracy
            optimal_threshold, optimal_value, optimized_metrics = find_optimal_threshold(df, metric='accuracy',
                                                                                         calibrated_threshold=calibrated_threshold)
        else:
            # For two-class, optimize F1 score
            optimal_threshold, optimal_value, optimized_metrics = find_optimal_threshold(df, metric='f1',
                                                                                         calibrated_threshold=calibrated_threshold)

        # Calculate threshold difference
        threshold_diff = None
        if optimal_threshold is not None:
            threshold_diff = optimal_threshold - calibrated_threshold

        if calibrated_metrics['is_single_class']:
            print(f"  ⚠ Model: {model_name}, Epoch: {epoch_num}, Dataset: {dataset_name}")
            print(f"     Calibrated - Brier: {calibrated_metrics['brier_score']:.4f}, Threshold: {_fmt_threshold(calibrated_threshold)}")
            if optimal_threshold is not None:
                print(f"     Optimized  - Brier: {optimized_metrics['brier_score']:.4f}, Threshold: {_fmt_threshold(optimal_threshold)}, Diff: {threshold_diff:+.{THRESHOLD_DECIMALS}f}")
        else:
            print(f"  ✓ Model: {model_name}, Epoch: {epoch_num}, Dataset: {dataset_name}")
            print(f"     Calibrated - F1: {calibrated_metrics['f1']:.4f}, Brier: {calibrated_metrics['brier_score']:.4f}, Threshold: {_fmt_threshold(calibrated_threshold)}")
            if optimal_threshold is not None:
                print(f"     Optimized  - F1: {optimized_metrics['f1']:.4f}, Brier: {optimized_metrics['brier_score']:.4f}, Threshold: {_fmt_threshold(optimal_threshold)}, Diff: {threshold_diff:+.{THRESHOLD_DECIMALS}f}")

        return model_name, epoch_num, dataset_name, calibrated_metrics, optimized_metrics, threshold_diff

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None, None, None, None, None, None



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
    Create ranking tables based on both calibrated and dataset-optimized thresholds.
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return

    csv_files = find_csv_files_recursively(directory)

    if not csv_files:
        print(f"No CSV files found in directory: {directory}")
        return

    print(f"Found {len(csv_files)} CSV file(s) in directory tree: {directory}")
    print("Checking files for required columns...\n")

    # Store results for both calibrated and optimized thresholds
    calibrated_results = defaultdict(lambda: defaultdict(dict))
    optimized_results = defaultdict(lambda: defaultdict(dict))
    threshold_diffs = defaultdict(lambda: defaultdict(dict))

    processed_count = 0
    skipped_count = 0

    for csv_filepath in csv_files:
        print(f"Processing {csv_filepath}...")

        # Check if file has required columns
        if check_csv_columns(csv_filepath):
            model_name, epoch_num, dataset_name, cal_metrics, opt_metrics, threshold_diff = process_csv_for_optimized_metrics(csv_filepath)

            if model_name and epoch_num is not None and dataset_name:
                if cal_metrics:
                    calibrated_results[(model_name, epoch_num)][dataset_name] = cal_metrics
                if opt_metrics:
                    optimized_results[(model_name, epoch_num)][dataset_name] = opt_metrics
                if threshold_diff is not None:
                    threshold_diffs[(model_name, epoch_num)][dataset_name] = threshold_diff
                processed_count += 1
            else:
                skipped_count += 1
        else:
            print(f"  ✗ File does not have required columns - skipping...")
            skipped_count += 1

    print(f"\n{'='*80}")
    print(f"Total CSV files found: {len(csv_files)}")
    print(f"Files processed: {processed_count}")
    print(f"Files skipped: {skipped_count}")
    print(f"{'='*80}\n")

    # Create ranking tables for both calibrated and optimized
    if calibrated_results:
        print("\n" + "="*80)
        print("CALIBRATED THRESHOLD RESULTS")
        print("="*80)
        create_ranking_table(calibrated_results, directory, suffix='calibrated')

        if optimized_results:
            print("\n" + "="*80)
            print("DATASET-OPTIMIZED THRESHOLD RESULTS")
            print("="*80)
            create_ranking_table(optimized_results, directory, suffix='optimized')

            # Create threshold comparison table
            create_threshold_comparison_table(threshold_diffs, calibrated_results, optimized_results, directory)
    else:
        print("No valid results to create ranking table.")


def create_ranking_table(results, output_dir, suffix=''):
    """
    Create and save ranking tables based on average Brier Score (lower is better).

    Args:
        results: Dictionary of results to process
        output_dir: Directory to save output files
        suffix: Optional suffix for output filenames (e.g., 'calibrated', 'optimized')
    """
    # Calculate average Brier Score for each model/epoch combination
    rankings = []

    for (model_name, epoch_num), datasets in results.items():
        # Collect all metrics
        brier_scores = [metrics['brier_score'] for metrics in datasets.values()]
        f1_scores = [metrics['f1'] for metrics in datasets.values() if metrics['f1'] is not None]
        all_accuracies = [metrics['accuracy'] for metrics in datasets.values()]
        precision_scores = [metrics['precision'] for metrics in datasets.values() if metrics['precision'] is not None]
        recall_scores = [metrics['recall'] for metrics in datasets.values() if metrics['recall'] is not None]

        # Count dataset types
        num_two_class = len(f1_scores)
        num_single_class = len(all_accuracies) - num_two_class
        num_total = len(all_accuracies)

        # Calculate averages
        avg_brier = np.mean(brier_scores) if brier_scores else None
        avg_f1 = np.mean(f1_scores) if f1_scores else None
        avg_accuracy = np.mean(all_accuracies) if all_accuracies else None
        avg_precision = np.mean(precision_scores) if precision_scores else None
        avg_recall = np.mean(recall_scores) if recall_scores else None

        # Calculate weighted Brier Score (weighted by sample count)
        sample_counts = [metrics['total_samples'] for metrics in datasets.values()]
        total_samples = sum(sample_counts)
        weighted_brier = None
        if brier_scores and total_samples > 0:
            weighted_brier = sum(b * s for b, s in zip(brier_scores, sample_counts)) / total_samples

        # Calculate weighted accuracy (weighted by sample count)
        # This is better than average accuracy as larger datasets contribute more
        weighted_accuracy = None
        if all_accuracies and total_samples > 0:
            weighted_accuracy = sum(a * s for a, s in zip(all_accuracies, sample_counts)) / total_samples

        # Calculate weighted F1 (weighted by sample count, only two-class datasets)
        weighted_f1 = None
        if f1_scores:
            two_class_sample_counts = [metrics['total_samples'] for metrics in datasets.values() if metrics['f1'] is not None]
            two_class_total = sum(two_class_sample_counts)
            if two_class_total > 0:
                weighted_f1 = sum(f * s for f, s in zip(f1_scores, two_class_sample_counts)) / two_class_total

        # Calculate Brier Score statistics
        min_brier = np.min(brier_scores) if brier_scores else None
        max_brier = np.max(brier_scores) if brier_scores else None
        median_brier = np.median(brier_scores) if brier_scores else None

        # Calculate F1 statistics (only for two-class datasets)
        min_f1 = np.min(f1_scores) if f1_scores else None
        max_f1 = np.max(f1_scores) if f1_scores else None
        median_f1 = np.median(f1_scores) if f1_scores else None

        # Calculate threshold statistics (all datasets)
        thresholds = [metrics['threshold'] for metrics in datasets.values()]
        median_threshold = np.median(thresholds) if thresholds else None
        min_threshold = np.min(thresholds) if thresholds else None
        max_threshold = np.max(thresholds) if thresholds else None

        rankings.append({
            'Rank': 0,  # Will be assigned after sorting
            'Model': model_name,
            'Epoch': epoch_num,
            'Avg_Brier': avg_brier,
            'Weighted_Brier': weighted_brier,
            'Median_Brier': median_brier,
            'Min_Brier': min_brier,
            'Max_Brier': max_brier,
            'Avg_F1': avg_f1,
            'Weighted_F1': weighted_f1,
            'Median_F1': median_f1,
            'Min_F1': min_f1,
            'Max_F1': max_f1,
            'Avg_Accuracy': avg_accuracy,
            'Weighted_Accuracy': weighted_accuracy,
            'Avg_Precision': avg_precision,
            'Avg_Recall': avg_recall,
            'Median_Threshold': median_threshold,
            'Min_Threshold': min_threshold,
            'Max_Threshold': max_threshold,
            'Total_Samples': total_samples,
            'Num_Total_Datasets': num_total,
            'Num_Two_Class': num_two_class,
            'Num_Single_Class': num_single_class,
            'datasets_detail': datasets  # Keep for detailed table
        })

    # Sort by weighted Brier Score (ascending - lower is better)
    # Falls back to average Brier if weighted is not available
    rankings.sort(key=lambda x: (x['Weighted_Brier'] if x['Weighted_Brier'] is not None else x['Avg_Brier'] if x['Avg_Brier'] is not None else float('inf')))

    # Assign ranks
    for i, entry in enumerate(rankings, start=1):
        entry['Rank'] = i

    # Create summary DataFrame
    summary_df = pd.DataFrame([
        {
            'Rank': r['Rank'],
            'Model': r['Model'],
            'Epoch': r['Epoch'],
            'Weighted_Brier': f"{r['Weighted_Brier']:.4f}" if r['Weighted_Brier'] is not None else 'N/A',
            'Avg_Brier': f"{r['Avg_Brier']:.4f}" if r['Avg_Brier'] is not None else 'N/A',
            'Median_Brier': f"{r['Median_Brier']:.4f}" if r['Median_Brier'] is not None else 'N/A',
            'Min_Brier': f"{r['Min_Brier']:.4f}" if r['Min_Brier'] is not None else 'N/A',
            'Max_Brier': f"{r['Max_Brier']:.4f}" if r['Max_Brier'] is not None else 'N/A',
            'Weighted_Accuracy': f"{r['Weighted_Accuracy']:.4f}" if r['Weighted_Accuracy'] is not None else 'N/A',
            'Avg_Accuracy': f"{r['Avg_Accuracy']:.4f}" if r['Avg_Accuracy'] is not None else 'N/A',
            'Weighted_F1': f"{r['Weighted_F1']:.4f}" if r['Weighted_F1'] is not None else 'N/A',
            'Avg_F1': f"{r['Avg_F1']:.4f}" if r['Avg_F1'] is not None else 'N/A',
            'Median_F1': f"{r['Median_F1']:.4f}" if r['Median_F1'] is not None else 'N/A',
            'Avg_Precision': f"{r['Avg_Precision']:.4f}" if r['Avg_Precision'] is not None else 'N/A',
            'Avg_Recall': f"{r['Avg_Recall']:.4f}" if r['Avg_Recall'] is not None else 'N/A',
            'Median_Threshold': f"{r['Median_Threshold']:.{THRESHOLD_DECIMALS}f}" if r['Median_Threshold'] is not None else 'N/A',
            'Total_Samples': r['Total_Samples'],
            'Total_Datasets': r['Num_Total_Datasets'],
            'Two_Class': r['Num_Two_Class'],
            'Single_Class': r['Num_Single_Class']
        }
        for r in rankings
    ])

    # Create detailed DataFrame (one row per model/epoch/dataset combination)
    detailed_rows = []
    for entry in rankings:
        model_name = entry['Model']
        epoch_num = entry['Epoch']
        rank = entry['Rank']

        for dataset_name, metrics in entry['datasets_detail'].items():
            detailed_rows.append({
                'Rank': rank,
                'Model': model_name,
                'Epoch': epoch_num,
                'Dataset': dataset_name,
                'Is_Single_Class': 'Yes' if metrics['is_single_class'] else 'No',
                'Brier_Score': f"{metrics['brier_score']:.4f}",
                'F1': f"{metrics['f1']:.4f}" if metrics['f1'] is not None else 'N/A',
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}" if metrics['precision'] is not None else 'N/A',
                'Recall': f"{metrics['recall']:.4f}" if metrics['recall'] is not None else 'N/A',
                'Threshold': _fmt_threshold(metrics['threshold']),

                'TP': metrics['tp'],
                'TN': metrics['tn'],
                'FP': metrics['fp'],
                'FN': metrics['fn'],
                'Total_Samples': metrics['total_samples']
            })

    detailed_df = pd.DataFrame(detailed_rows)

    # Save to CSV files with optional suffix
    suffix_str = f'_{suffix}' if suffix else ''
    summary_path = os.path.join(output_dir, f'model_ranking_summary{suffix_str}.csv')
    detailed_path = os.path.join(output_dir, f'model_ranking_detailed{suffix_str}.csv')

    summary_df.to_csv(summary_path, index=False)
    detailed_df.to_csv(detailed_path, index=False)

    print(f"\n{'='*80}")
    print("MODEL RANKING BY WEIGHTED BRIER SCORE (Lower is Better)")
    print("Weighted by sample count - larger datasets have more influence")
    print(f"{'='*80}\n")
    print(summary_df.to_string(index=False))
    print(f"\n{'='*80}")
    print(f"Summary table saved to: {summary_path}")
    print(f"Detailed table saved to: {detailed_path}")
    print(f"{'='*80}\n")


def create_threshold_comparison_table(threshold_diffs, calibrated_results, optimized_results, output_dir):
    """
    Create a table comparing calibrated vs optimized thresholds.

    Args:
        threshold_diffs: Dictionary of threshold differences
        calibrated_results: Dictionary of calibrated metrics
        optimized_results: Dictionary of optimized metrics
        output_dir: Directory to save the comparison table
    """
    comparison_rows = []

    for (model_name, epoch_num), datasets in threshold_diffs.items():
        for dataset_name, threshold_diff in datasets.items():
            cal_metrics = calibrated_results[(model_name, epoch_num)][dataset_name]
            opt_metrics = optimized_results[(model_name, epoch_num)][dataset_name]

            # Calculate improvements using raw numeric values (not formatted strings)
            f1_improvement = None
            if opt_metrics['f1'] is not None and cal_metrics['f1'] is not None:
                f1_improvement = opt_metrics['f1'] - cal_metrics['f1']

            brier_improvement = cal_metrics['brier_score'] - opt_metrics['brier_score']

            comparison_rows.append({
                'Model': model_name,
                'Epoch': epoch_num,
                'Dataset': dataset_name,
                'Is_Single_Class': 'Yes' if cal_metrics['is_single_class'] else 'No',
                'Calibrated_Threshold': cal_metrics['threshold'],
                'Optimized_Threshold': opt_metrics['threshold'],
                'Threshold_Diff': threshold_diff,
                'Calibrated_F1': cal_metrics['f1'] if cal_metrics['f1'] is not None else None,
                'Optimized_F1': opt_metrics['f1'] if opt_metrics['f1'] is not None else None,
                'F1_Improvement': f1_improvement,
                'Calibrated_Brier': cal_metrics['brier_score'],
                'Optimized_Brier': opt_metrics['brier_score'],
                'Brier_Improvement': brier_improvement
            })

    comparison_df = pd.DataFrame(comparison_rows)

    # Format numeric columns for display and CSV output
    display_df = comparison_df.copy()
    display_df['Calibrated_Threshold'] = display_df['Calibrated_Threshold'].apply(lambda x: f"{x:.{THRESHOLD_DECIMALS}f}")
    display_df['Optimized_Threshold'] = display_df['Optimized_Threshold'].apply(lambda x: f"{x:.{THRESHOLD_DECIMALS}f}")
    display_df['Threshold_Diff'] = display_df['Threshold_Diff'].apply(lambda x: f"{x:+.{THRESHOLD_DECIMALS}f}")
    display_df['Calibrated_F1'] = display_df['Calibrated_F1'].apply(lambda x: f"{x:.4f}" if x is not None else 'N/A')
    display_df['Optimized_F1'] = display_df['Optimized_F1'].apply(lambda x: f"{x:.4f}" if x is not None else 'N/A')
    display_df['F1_Improvement'] = display_df['F1_Improvement'].apply(lambda x: f"{x:+.4f}" if x is not None else 'N/A')
    display_df['Calibrated_Brier'] = display_df['Calibrated_Brier'].apply(lambda x: f"{x:.4f}")
    display_df['Optimized_Brier'] = display_df['Optimized_Brier'].apply(lambda x: f"{x:.4f}")
    display_df['Brier_Improvement'] = display_df['Brier_Improvement'].apply(lambda x: f"{x:+.4f}")

    comparison_path = os.path.join(output_dir, 'threshold_comparison.csv')
    display_df.to_csv(comparison_path, index=False)

    print(f"\n{'='*80}")
    print("THRESHOLD COMPARISON: Calibrated vs Dataset-Optimized")
    print(f"{'='*80}\n")
    print(display_df.to_string(index=False))
    print(f"\n{'='*80}")
    print(f"Comparison table saved to: {comparison_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    process_csv_files(directory)
