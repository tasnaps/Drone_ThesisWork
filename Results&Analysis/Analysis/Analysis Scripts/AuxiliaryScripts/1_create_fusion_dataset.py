import pandas as pd
import os
import sys
import re
import shutil
from datetime import datetime
from pathlib import Path


def _is_epoch_segment(seg: str) -> bool:
    """Return True if the path segment represents an epoch folder name.
    Matches forms like '5Epoch', 'Epoch-5', 'Epoch5', '5-Epoch', '1Epoch', 'Epoch_10', etc.
    """
    if not seg:
        return False
    return re.fullmatch(r"(?i:(?:\d+[-_]?Epoch|Epoch[-_]?\d+))", seg) is not None


def _extract_epoch_num(seg: str) -> int | None:
    """Extract the integer epoch number from an epoch segment name."""
    if not seg:
        return None
    m = re.match(r"(?i)(?:([0-9]+)[-_]?Epoch|Epoch[-_]?([0-9]+))", seg)
    if not m:
        return None
    num = m.group(1) or m.group(2)
    try:
        return int(num)
    except Exception:
        return None


def parse_model_epoch_epochpath(full_path: str):
    """Given a full CSV path, return (model_name, epoch_num, epoch_folder_path).
    Uses pathlib to robustly handle Windows absolute paths.
    epoch_folder_path is returned as a Path object (absolute).
    Returns (None, None, None) if cannot parse.
    """
    try:
        p = Path(full_path)
        parts = p.parts  # e.g. ('Z:\\', 'Experiment', 'Evaluation', 'Model', '5Epoch', 'Fusion', 'file.csv')
        # Find index of epoch segment
        epoch_idx = None
        for i, seg in enumerate(parts):
            # seg could be like 'Z:\\' for drive+root; skip those
            seg_clean = seg.rstrip('\\/')
            if _is_epoch_segment(seg_clean):
                epoch_idx = i
                break
        if epoch_idx is None or epoch_idx == 0:
            return None, None, None
        model_name = parts[epoch_idx - 1].rstrip('\\/')
        epoch_seg = parts[epoch_idx].rstrip('\\/')
        epoch_num = _extract_epoch_num(epoch_seg)
        # Build epoch folder path including drive/root
        epoch_folder_path = Path(*parts[:epoch_idx + 1])
        if not epoch_folder_path.is_absolute():
            # If for some reason it's not absolute, anchor with drive of original path
            epoch_folder_path = p.drive and Path(p.drive + os.sep) / epoch_folder_path or epoch_folder_path
        return model_name, epoch_num, epoch_folder_path
    except Exception:
        return None, None, None


def analyze_path(full_path: str):
    """Hardcoded / robust path analysis returning (model_name, epoch_num, epoch_folder_str, layout_type).
    layout_type:
      - 'clustered' => detailed files all reside in a single 'datasets' folder per epoch
      - 'distributed' => each dataset has its own subfolder under the epoch folder
    Assumptions based on user description:
      * wav2vec2-no-augment / wav2vec2Augments use clustered layout (have 'datasets')
      * cnn_lstm_evaluation_series_* and Resnet_Results_* use distributed layout (no 'datasets', dataset folders under epoch)
    """
    p = Path(full_path)
    parts = list(p.parts)
    # Find epoch segment
    epoch_idx = None
    for i, seg in enumerate(parts):
        sclean = seg.replace('-', '').replace('_', '')
        if re.fullmatch(r"(?i)(\d+Epoch|Epoch\d+)", sclean):
            epoch_idx = i
            break
    if epoch_idx is None or epoch_idx == 0:
        return None, None, None, None
    model_name = parts[epoch_idx - 1]
    epoch_seg = parts[epoch_idx]
    # Extract number
    m = re.search(r"(\d+)", epoch_seg)
    epoch_num = int(m.group(1)) if m else None
    # Build epoch folder path as a normalized string for grouping
    epoch_folder_path = Path(*parts[:epoch_idx + 1])
    epoch_folder_str = str(epoch_folder_path)
    # Determine layout
    layout_type = 'clustered' if 'datasets' in [seg.lower() for seg in parts] else 'distributed'
    # Override: if model_name clearly indicates cnn_lstm or Resnet => distributed
    if re.search(r"cnn_lstm|resnet", model_name, re.IGNORECASE):
        layout_type = 'distributed'
    return model_name, epoch_num, epoch_folder_str, layout_type


def find_detailed_csv_files(root_directory):
    """Discover detailed csv files and attach model/epoch/layout metadata.
    Searches for both:
    - *_detailed_files.csv (wav2vec2, cnn_lstm)
    - *_results_*.csv (Resnet)
    """
    detailed_files = []  # list of dict rows
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            # Skip files we don't care about
            if not (file.lower().endswith('_detailed_files.csv') or
                    (file.lower().endswith('.csv') and '_results_' in file.lower())):
                continue

            # Skip metrics files
            if '_metrics.txt' in file.lower():
                continue

            full_path = os.path.join(root, file)

            # Extract dataset name from filename
            if '_detailed_files.csv' in file.lower():
                dataset_name = file.replace('_detailed_files.csv', '')
            elif '_results_' in file.lower():
                # For Resnet: DronePrint_results_20251111_100611.csv -> DronePrint
                dataset_name = file.split('_results_')[0]
            else:
                continue

            model_name, epoch_num, epoch_folder_str, layout_type = analyze_path(full_path)
            if model_name is None or epoch_num is None:
                print(f"  ⚠ Skip (cannot parse model/epoch): {full_path}")
                continue
            detailed_files.append({
                'model': model_name,
                'epoch': epoch_num,
                'epoch_dir': epoch_folder_str,
                'layout': layout_type,
                'dataset': dataset_name,
                'path': full_path
            })
    return detailed_files


def check_required_columns(df):
    """
    Check if DataFrame has all required columns.
    """
    required_columns = {
        'file_id', 'true_label', 'predicted_label', 'drone_probability',
        'aggregation_method', 'aggregation_threshold', 'split'
    }
    return required_columns.issubset(set(df.columns))


def backup_existing_fusion_file(fusion_path, model_name, epoch_num, backup_dir):
    """
    Backup an existing Fusion file to the backup directory with proper naming.

    Args:
        fusion_path: Path to the existing Fusion file
        model_name: Name of the model
        epoch_num: Epoch number
        backup_dir: Directory to save backups

    Returns:
        str: Path to the backed up file, or None if no backup was needed
    """
    if not os.path.exists(fusion_path):
        return None

    # Create backup directory if it doesn't exist
    os.makedirs(backup_dir, exist_ok=True)

    # Create backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{model_name}_Epoch{epoch_num}_Fusion_detailed_files_backup_{timestamp}.csv"
    backup_path = os.path.join(backup_dir, backup_filename)

    # Move the file to backup location
    shutil.move(fusion_path, backup_path)

    return backup_path


def create_fusion_datasets(root_directory, output_dir=None, backup_dir=None):
    """
    Create Fusion datasets by combining all datasets except CalibrationDataset and existing Fusion.
    Creates one fusion file per model/epoch combination.
    Backs up existing Fusion files before creating new ones.

    Args:
        root_directory: Root directory to search for CSV files
        output_dir: Optional output directory. If None, saves alongside original files.
        backup_dir: Directory to backup existing Fusion files (default: C:\\Users\\tapio\\Desktop\\backups\\OriginalFusionDatasets)
    """
    if backup_dir is None:
        backup_dir = r"C:\Users\tapio\Desktop\backups\OriginalFusionDatasets"
    print(f"Searching for detailed CSV files in: {root_directory}")
    print("="*80)

    rows = find_detailed_csv_files(root_directory)
    if not rows:
        print("No detailed CSV files found!")
        return
    # Build dataset list (names only)
    all_dataset_names = sorted({r['dataset'] for r in rows})
    print(f"Found datasets (unique names): {all_dataset_names}")
    # Exclusions
    exclude_names = {'CalibrationDataset', 'Fusion', 'fusion'}
    rows = [r for r in rows if r['dataset'] not in exclude_names and r['dataset'].lower() != 'fusion']

    # Group by (model, epoch, epoch_folder_str, layout)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r['model'], r['epoch'], r['epoch_dir'], r['layout'])].append(r)

    print(f"Model/Epoch combinations detected: {len(groups)}")
    for key, items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        model, epoch, epoch_dir_str, layout = key
        ds_names = sorted({i['dataset'] for i in items})
        print(f"  - {model} epoch {epoch} layout={layout} datasets={len(ds_names)} -> {ds_names}")

    fusion_created = 0

    for (model, epoch, epoch_dir_str, layout), items in groups.items():
        print(f"\nProcessing Fusion: model={model} epoch={epoch} layout={layout}")
        # Decide output path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = Path(output_dir) / f"{model}_Epoch{epoch}_Fusion_detailed_files.csv"
        else:
            epoch_dir = Path(epoch_dir_str)
            if layout == 'clustered':
                # Expect a 'datasets' folder under epoch_dir
                candidate = epoch_dir / 'datasets'
                out_path = (candidate / 'Fusion_detailed_files.csv') if candidate.exists() else (epoch_dir / 'Fusion_detailed_files.csv')
            else:
                # distributed => save directly in epoch_dir
                out_path = epoch_dir / 'Fusion_detailed_files.csv'
        print(f"  Output path: {out_path}")

        # Load datasets
        dfs = []
        included = []
        for rec in items:
            try:
                df = pd.read_csv(rec['path'])
                if not check_required_columns(df):
                    print(f"   ⚠ Missing columns: {rec['dataset']} -> skip")
                    continue
                df = df.copy()
                df['source_dataset'] = rec['dataset']
                dfs.append(df)
                included.append(rec['dataset'])
                print(f"   ✓ {rec['dataset']} rows={len(df)}")
            except Exception as e:
                print(f"   ✗ Error reading {rec['path']}: {e}")
        if not dfs:
            print("  ⚠ No datasets loaded for this group; skipping Fusion creation.")
            continue
        fusion_df = pd.concat(dfs, ignore_index=True)
        fusion_df['file_id'] = range(len(fusion_df))
        fusion_df = fusion_df.sort_values('file_id').reset_index(drop=True)
        # Backup existing
        if out_path.exists():
            backup_path = backup_existing_fusion_file(str(out_path), model, epoch, backup_dir)
            if backup_path:
                print(f"  📦 Backup: {backup_path}")
        # Save
        fusion_df.to_csv(out_path, index=False)
        print(f"  ✓ Saved Fusion: {out_path}")
        print(f"  Stats: total={len(fusion_df)} label0={(fusion_df.true_label==0).sum()} label1={(fusion_df.true_label==1).sum()} threshold={fusion_df.aggregation_threshold.iloc[0]:.6f}")
        src_counts = fusion_df.source_dataset.value_counts().sort_index()
        for ds_name, cnt in src_counts.items():
            print(f"    - {ds_name}: {cnt}")
        fusion_created += 1
        print("  " + "-"*60)

    print("\n" + "="*80)
    print(f"SUMMARY: Created {fusion_created} Fusion dataset(s)")
    print("="*80)


def main():
    """
    Main function to run the fusion dataset creation.
    Usage: python create_fusion_dataset.py <root_directory> [output_directory] [backup_directory]
    """
    if len(sys.argv) < 2:
        print("Usage: python create_fusion_dataset.py <root_directory> [output_directory] [backup_directory]")
        print("\nExample:")
        print("  python create_fusion_dataset.py Z:\\Experiment\\Evaluation")
        print("  python create_fusion_dataset.py Z:\\Experiment\\Evaluation C:\\Output")
        print("  python create_fusion_dataset.py Z:\\Experiment\\Evaluation C:\\Output C:\\Backups")
        print("\nDefault backup directory: C:\\Users\\tapio\\Desktop\\backups\\OriginalFusionDatasets")
        return

    root_directory = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    backup_dir = sys.argv[3] if len(sys.argv) > 3 else r"C:\Users\tapio\Desktop\backups\OriginalFusionDatasets"

    if not os.path.exists(root_directory):
        print(f"Error: Directory '{root_directory}' does not exist!")
        return

    print(f"Backup directory: {backup_dir}")
    print("="*80)

    create_fusion_datasets(root_directory, output_dir, backup_dir)


if __name__ == "__main__":
    main()
