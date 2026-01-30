"""
Fix Threshold Values Script
============================
This script checks all *_detailed_files.csv files and fixes any that have
aggregation_threshold = 0.0, 0.001, or 0.01 by:
1. First trying to read the correct threshold from CalibrationDataset_detailed_files.csv
2. If that also has a bad threshold, recalculating the optimal threshold using the calibration data

The calibrated threshold is the same for all datasets in an epoch but different for every epoch.

Folder structure expected:
    Results/Model/Epoch/datasets/*_detailed_files.csv

Usage:
    python 0_fix_threshold_values.py <directory>
    python 0_fix_threshold_values.py "C:/Users/tapio/Desktop/Results23_01_2026"
"""

import pandas as pd
import numpy as np
import os
import sys
import glob
from collections import defaultdict

# Treat these as "bad/placeholder" thresholds that should be replaced.
# Many pipelines accidentally serialize a calibrated threshold with low precision
BAD_THRESHOLD_EXACT = {0.0, 0.001, 0.01}
BAD_THRESHOLD_EPS = 1e-9  # Epsilon for floating point comparison


def find_optimal_threshold(true_labels, probabilities):
    """
    Calculate the optimal threshold that maximizes F1 score.
    Uses scipy.optimize.minimize_scalar with Brent's method for efficient adaptive search.
    """
    from sklearn.metrics import f1_score
    from scipy.optimize import minimize_scalar

    def negative_f1(threshold):
        """Returns negative F1 (since we minimize)."""
        predictions = (probabilities >= threshold).astype(int)
        return -f1_score(true_labels, predictions, zero_division=0)

    # Use Brent's method - it adaptively narrows down the search
    # Search in range [0.02, 0.98] to avoid edge cases
    result = minimize_scalar(negative_f1, bounds=(0.02, 0.98), method='bounded')

    best_threshold = result.x
    best_f1 = -result.fun

    # Round to 10 decimal places for clean output
    best_threshold = round(best_threshold, 12)

    return best_threshold, best_f1


def _to_float_or_none(x):
    """Best-effort float parsing for values that might be '', '0.0', '0.0010000', etc."""
    try:
        if x is None:
            return None
        if isinstance(x, float) and np.isnan(x):
            return None
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return None
        return float(s)
    except Exception:
        return None


def _is_bad_threshold_value(v: float | None) -> bool:
    if v is None:
        return False
    if abs(v) <= BAD_THRESHOLD_EPS:
        return True
    # Treat anything extremely close to any bad threshold value as bad.
    if any(abs(v - t) <= BAD_THRESHOLD_EPS for t in BAD_THRESHOLD_EXACT):
        return True
    return False


def find_calibration_file(epoch_dir):
    """
    Find CalibrationDataset_detailed_files.csv in the epoch directory.
    Searches multiple possible locations based on folder structure.
    """
    search_patterns = [
        # Direct in epoch folder
        os.path.join(epoch_dir, 'CalibrationDataset_detailed_files.csv'),
        os.path.join(epoch_dir, 'Calibration*_detailed_files.csv'),
        # In datasets subfolder
        os.path.join(epoch_dir, 'datasets', 'CalibrationDataset_detailed_files.csv'),
        os.path.join(epoch_dir, 'datasets', 'Calibration*_detailed_files.csv'),
        # In CalibrationDataset subfolder
        os.path.join(epoch_dir, 'CalibrationDataset', '*_detailed_files.csv'),
        os.path.join(epoch_dir, 'Calibration*', '*_detailed_files.csv'),
    ]

    for pattern in search_patterns:
        matches = glob.glob(pattern)
        for match in matches:
            if os.path.exists(match):
                return match

    return None


def get_threshold_from_calibration_file(calibration_file):
    """
    Extract or calculate threshold from calibration file.
    Returns (threshold, source) where threshold is a float and source describes origin.

    Important: we always return a numeric float so downstream writing is consistent.
    """
    try:
        # Read without converting types to preserve exact string representation
        df = pd.read_csv(calibration_file, dtype=str)

        # First try to extract existing threshold
        if 'aggregation_threshold' in df.columns:
            threshold_values = df['aggregation_threshold'].dropna()
            if len(threshold_values) > 0:
                threshold_str = threshold_values.iloc[0]
                existing_threshold = _to_float_or_none(threshold_str)
                if existing_threshold is not None and not _is_bad_threshold_value(existing_threshold):
                    return float(existing_threshold), 'extracted'

        # If threshold is bad (0.0/0.001/0.01) or missing, calculate it from the data
        # Re-read with proper types for calculation
        df_calc = pd.read_csv(calibration_file)
        if 'true_label' in df_calc.columns and 'drone_probability' in df_calc.columns:
            true_labels = df_calc['true_label'].values
            probabilities = df_calc['drone_probability'].values

            # Check if we have both classes
            unique_labels = np.unique(true_labels)
            if len(unique_labels) >= 2:
                calculated_threshold, f1 = find_optimal_threshold(true_labels, probabilities)
                return float(calculated_threshold), f'calculated (F1={f1:.4f})'
            else:
                # Only one class - use median probability as threshold
                calculated_threshold = np.median(probabilities)
                return float(calculated_threshold), 'calculated (single class - used median)'

        return None, 'failed'

    except Exception as e:
        print(f"    Error reading calibration file: {e}")
        return None, f'error: {e}'


def find_all_detailed_csv_files(directory):
    """
    Recursively find all *_detailed_files.csv files.
    """
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('_detailed_files.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files


def extract_epoch_dir(csv_filepath):
    """
    Extract the epoch directory from a CSV file path.
    Handles patterns like:
    - .../Model/100Epoch/Dataset/file.csv -> .../Model/100Epoch
    - .../Model/Epoch1/datasets/file.csv -> .../Model/Epoch1
    """
    import re

    path_parts = csv_filepath.split(os.sep)

    for i, part in enumerate(path_parts):
        # Match patterns like "100Epoch", "10Epoch", "Epoch1", "Epoch10"
        if re.match(r'^(\d+)epoch$|^epoch(\d+)$', part, re.IGNORECASE):
            # Return path up to and including the epoch folder
            return os.sep.join(path_parts[:i+1])

    return None


def check_threshold_value(csv_filepath):
    """
    Check if a CSV file has a bad threshold value (0.0, 0.001, or 0.01).
    Returns (needs_fix, current_threshold)
    """
    try:
        # Read as strings to detect values like '0.01000000000000000'
        df = pd.read_csv(csv_filepath, nrows=5, dtype=str)

        if 'aggregation_threshold' in df.columns:
            threshold_values = df['aggregation_threshold'].dropna()
            if len(threshold_values) > 0:
                threshold = _to_float_or_none(threshold_values.iloc[0])
                if _is_bad_threshold_value(threshold):
                    return True, threshold

        return False, None

    except Exception:
        return False, None


def update_csv_threshold(csv_filepath, new_threshold):
    """
    Update the aggregation_threshold column in a CSV file with the new threshold value.

    Write as numeric float (same value on every row).
    Uses fixed decimal notation (not scientific notation).
    """
    try:
        df = pd.read_csv(csv_filepath)

        if 'aggregation_threshold' in df.columns:
            threshold_float = _to_float_or_none(new_threshold)
            if threshold_float is None:
                print(f"    Error: computed threshold is None for {os.path.basename(csv_filepath)}")
                return False

            df['aggregation_threshold'] = float(threshold_float)

            # Use fixed decimal format to avoid scientific notation
            df.to_csv(csv_filepath, index=False, float_format='%.15f')
            return True
        else:
            print(f"    Warning: No aggregation_threshold column in {os.path.basename(csv_filepath)}")
            return False

    except Exception as e:
        print(f"    Error updating {csv_filepath}: {e}")
        return False


def fix_threshold_values(directory):
    """
    Main function to find and fix all CSV files with 0.0/0.01 thresholds.
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return

    print(f"Scanning directory: {directory}")
    print("=" * 60)

    # Find all detailed CSV files
    all_csv_files = find_all_detailed_csv_files(directory)
    print(f"Found {len(all_csv_files)} *_detailed_files.csv files")

    # Group files by epoch directory
    epoch_files = defaultdict(list)
    files_without_epoch = []

    for csv_file in all_csv_files:
        epoch_dir = extract_epoch_dir(csv_file)
        if epoch_dir:
            epoch_files[epoch_dir].append(csv_file)
        else:
            files_without_epoch.append(csv_file)

    print(f"Found {len(epoch_files)} epoch directories")
    print()

    # Track statistics
    total_files_checked = 0
    files_with_zero_threshold = 0
    files_fixed = 0
    files_failed = 0

    # Process each epoch directory
    for epoch_dir in sorted(epoch_files.keys()):
        csv_files = epoch_files[epoch_dir]

        # Check if any files in this epoch have 0.0 threshold
        files_needing_fix = []
        for csv_file in csv_files:
            has_zero, current = check_threshold_value(csv_file)
            total_files_checked += 1
            if has_zero:
                files_needing_fix.append(csv_file)
                files_with_zero_threshold += 1

        if not files_needing_fix:
            continue

        # Get relative epoch path for display
        rel_epoch = os.path.relpath(epoch_dir, directory)
        print(f"\n📁 {rel_epoch}")
        print(f"   Found {len(files_needing_fix)} file(s) with bad threshold (0/0.001/0.01)")

        # Find calibration file and get threshold
        calibration_file = find_calibration_file(epoch_dir)

        if calibration_file:
            print(f"   📄 Calibration file: {os.path.basename(calibration_file)}")
            threshold, source = get_threshold_from_calibration_file(calibration_file)

            if threshold is not None:
                print(f"   ✓ Threshold: {float(threshold):.17g} ({source})")

                # Update all files that need fixing
                for csv_file in files_needing_fix:
                    filename = os.path.basename(csv_file)
                    if update_csv_threshold(csv_file, threshold):
                        print(f"      ✓ Fixed: {filename}")
                        files_fixed += 1
                    else:
                        print(f"      ✗ Failed: {filename}")
                        files_failed += 1
            else:
                print(f"   ✗ Could not determine threshold ({source})")
                files_failed += len(files_needing_fix)
        else:
            print(f"   ✗ No calibration file found in epoch directory")
            files_failed += len(files_needing_fix)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total CSV files checked: {total_files_checked}")
    print(f"Files with threshold = 0.0: {files_with_zero_threshold}")
    print(f"Files successfully fixed: {files_fixed}")
    print(f"Files failed to fix: {files_failed}")

    if files_without_epoch:
        print(f"\nNote: {len(files_without_epoch)} files were not in an epoch directory structure")


def preview_mode(directory):
    """
    Preview what would be fixed without making changes.
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return

    print(f"PREVIEW MODE - No changes will be made")
    print(f"Scanning directory: {directory}")
    print("=" * 60)

    all_csv_files = find_all_detailed_csv_files(directory)
    print(f"Found {len(all_csv_files)} *_detailed_files.csv files")

    epoch_files = defaultdict(list)

    for csv_file in all_csv_files:
        epoch_dir = extract_epoch_dir(csv_file)
        if epoch_dir:
            epoch_files[epoch_dir].append(csv_file)

    files_needing_fix = 0

    for epoch_dir in sorted(epoch_files.keys()):
        csv_files = epoch_files[epoch_dir]

        zero_files = []
        for csv_file in csv_files:
            has_zero, current = check_threshold_value(csv_file)
            if has_zero:
                zero_files.append(csv_file)
                files_needing_fix += 1

        if zero_files:
            rel_epoch = os.path.relpath(epoch_dir, directory)
            calibration_file = find_calibration_file(epoch_dir)

            print(f"\n📁 {rel_epoch}")
            print(f"   Files with 0.0 threshold: {len(zero_files)}")

            if calibration_file:
                threshold, source = get_threshold_from_calibration_file(calibration_file)
                if threshold:
                    # Display threshold - handle both string and numeric
                    if isinstance(threshold, str):
                        threshold_display = float(threshold)
                    else:
                        threshold_display = threshold
                    print(f"   Would use threshold: {threshold_display:.15f} ({source})")
                else:
                    print(f"   ⚠ Could not determine threshold")
            else:
                print(f"   ⚠ No calibration file found")

            for f in zero_files:
                print(f"      - {os.path.basename(f)}")

    print(f"\n{'=' * 60}")
    print(f"Total files that would be fixed: {files_needing_fix}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python 0_fix_threshold_values.py <directory>")
        print("  python 0_fix_threshold_values.py <directory> --preview")
        print()
        print("Example:")
        print('  python 0_fix_threshold_values.py "C:/Users/tapio/Desktop/Results23_01_2026"')
        sys.exit(1)

    directory = sys.argv[1]

    if len(sys.argv) > 2 and sys.argv[2] == '--preview':
        preview_mode(directory)
    else:
        fix_threshold_values(directory)
