import os
import sys
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List
import argparse

# Add sklearn metrics for PR-AUC and ROC-AUC
try:
    from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    # Fallback minimal implementations
    def precision_recall_curve(y_true, probas):
        # Sort by probas descending
        order = np.argsort(-probas)
        y_true_sorted = y_true[order]
        probas_sorted = probas[order]
        tp = 0
        fp = 0
        precisions = []
        recalls = []
        thresholds = []
        pos_total = y_true.sum()
        if pos_total == 0:
            return np.array([1.0]), np.array([0.0]), np.array([])
        last_score = None
        for yt, score in zip(y_true_sorted, probas_sorted):
            if score != last_score:
                if tp + fp > 0:
                    precisions.append(tp / (tp + fp))
                    recalls.append(tp / pos_total)
                    thresholds.append(score)
                last_score = score
            if yt == 1:
                tp += 1
            else:
                fp += 1
        # Append final point
        if tp + fp > 0:
            precisions.append(tp / (tp + fp))
            recalls.append(tp / pos_total)
        return np.array(recalls), np.array(precisions), np.array(thresholds)

    def average_precision_score(y_true, probas):
        # Basic AP approximation (step-wise)
        order = np.argsort(-probas)
        y_true_sorted = y_true[order]
        tp_cum = np.cumsum(y_true_sorted)
        total_pos = y_true.sum()
        if total_pos == 0:
            return np.nan
        precisions = tp_cum / (np.arange(len(y_true_sorted)) + 1)
        recalls = tp_cum / total_pos
        # Sklearn style interpolation: sum over (R_n - R_{n-1}) * P_n where y_true_sorted[i]==1
        ap = 0.0
        last_recall = 0.0
        for p, r, y in zip(precisions, recalls, y_true_sorted):
            if y == 1:
                ap += p * (r - last_recall)
                last_recall = r
        return ap

    def roc_curve(y_true, probas):
        # Simple ROC curve implementation
        order = np.argsort(-probas)
        y_true_sorted = y_true[order]
        prob_sorted = probas[order]
        tp = 0
        fp = 0
        pos = y_true.sum()
        neg = (y_true == 0).sum()
        if pos == 0 or neg == 0:
            return np.array([0,1]), np.array([0,1]), np.array([1,0])
        tprs = [0.0]
        fprs = [0.0]
        thresholds = [np.inf]
        prev_score = None
        for yt, score in zip(y_true_sorted, prob_sorted):
            if score != prev_score:
                tprs.append(tp / pos)
                fprs.append(fp / neg)
                thresholds.append(score)
                prev_score = score
            if yt == 1:
                tp += 1
            else:
                fp += 1
        tprs.append(tp / pos)
        fprs.append(fp / neg)
        thresholds.append(-np.inf)
        return np.array(fprs), np.array(tprs), np.array(thresholds)

    def roc_auc_score(y_true, probas):
        fpr, tpr, _ = roc_curve(y_true, probas)
        # Trapezoidal integration
        return float(np.trapz(tpr, fpr))

# Required columns per user specification
REQUIRED_COLUMNS = {
    'file_id', 'true_label', 'predicted_label', 'drone_probability',
    'aggregation_method', 'aggregation_threshold', 'split'
}

epoch_dir_pattern = re.compile(r'^(\d+)Epoch$', re.IGNORECASE)
GENERIC_PARENT_NAMES = {'datasets', 'dataset', 'data', 'csv', 'files', 'results'}


def find_csv_files_recursively(root: str) -> List[str]:
    csv_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith('.csv'):
                csv_files.append(os.path.join(dirpath, f))
    return csv_files


def csv_has_required_columns(path: str) -> bool:
    try:
        df_head = pd.read_csv(path, nrows=0)
        return REQUIRED_COLUMNS.issubset(set(df_head.columns))
    except Exception:
        return False


def extract_epoch_number(path: str, root: str) -> str:
    """Find the nearest ancestor directory whose name matches <digits>Epoch and return the digits as string.
    If none found, return 'Unknown'."""
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    # Walk upward from deepest directory containing the file toward root
    for i in range(len(parts)-1, -1, -1):
        m = epoch_dir_pattern.match(parts[i])
        if m:
            return m.group(1)  # digits only
    return 'Unknown'


def extract_dataset_name(csv_path: str) -> str:
    """Heuristics to derive dataset name.
    Cases:
      1. If parent folder looks like a dataset (not generic, not evaluation timestamp) -> use parent.
      2. If parent is generic (e.g. 'datasets') or evaluation folder -> derive from filename.
      3. Filenames like 'DronePrint_detailed_files.csv' -> take leading token before first underscore.
      4. Fallback to filename stem.
    """
    parent = os.path.basename(os.path.dirname(csv_path))
    fname = os.path.basename(csv_path)
    stem = os.path.splitext(fname)[0]

    # Patterns to detect evaluation/timestamp style folders
    if (parent.lower() in GENERIC_PARENT_NAMES) or re.search(r'evaluation_|eval_', parent, re.IGNORECASE) or re.search(r'\d{8}_\d{6}', parent):
        # Derive from filename
        if stem.endswith('_detailed_files'):
            core = stem[:-len('_detailed_files')]
        elif stem.endswith('_files'):
            core = stem[:-len('_files')]
        else:
            core = stem
        if '_' in core:
            return core.split('_')[0]
        return core

    # Parent might still be something like Fusion, Tapio, etc.
    return parent


def summarize_probabilities(df: pd.DataFrame) -> Dict[str, float]:
    """Return statistics for drone_probability column for provided subset df."""
    probs = df['drone_probability'].astype(float) if not df.empty else pd.Series(dtype=float)
    if probs.empty:
        return {
            'count': 0,
            'mean': np.nan,
            'min': np.nan,
            'max': np.nan,
            'median': np.nan,
            'std': np.nan,
            'q1': np.nan,
            'q3': np.nan,
            'ci95': np.nan,
        }
    count = probs.count()
    std = float(probs.std(ddof=0)) if count > 0 else np.nan
    ci95 = float(1.96 * std / np.sqrt(count)) if count > 0 else np.nan
    return {
        'count': int(count),
        'mean': float(probs.mean()),
        'min': float(probs.min()),
        'max': float(probs.max()),
        'median': float(probs.median()),
        'std': std,
        'q1': float(probs.quantile(0.25)),
        'q3': float(probs.quantile(0.75)),
        'ci95': ci95,
    }


def merge_stat_records(records: List[pd.DataFrame]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def pooled_std(n1, s1, n2, s2):
    if any(v is None for v in [n1, s1, n2, s2]):
        return np.nan
    if n1 < 2 or n2 < 2 or np.isnan(s1) or np.isnan(s2):
        return np.nan
    return np.sqrt(((n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)) / (n1 + n2 - 2))


def compute_pr_auc(df: pd.DataFrame) -> Dict[str, float]:
    """Compute average precision (PR-AUC) and return curve arrays for plotting.
    Returns dict with AP, prevalence, precision, recall arrays (as lists)."""
    if df.empty:
        return {'Average_Precision': np.nan, 'Positive_Prevalence': np.nan, 'precision': [], 'recall': []}
    y_true = df['true_label'].values
    y_score = df['drone_probability'].astype(float).values
    pos_total = (y_true == 1).sum()
    neg_total = (y_true == 0).sum()
    if pos_total == 0 or neg_total == 0:
        # Cannot form meaningful curve if only one class present
        prevalence = pos_total / (pos_total + neg_total) if (pos_total + neg_total) > 0 else np.nan
        return {'Average_Precision': np.nan, 'Positive_Prevalence': prevalence, 'precision': [], 'recall': []}
    ap = average_precision_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    prevalence = pos_total / (pos_total + neg_total)
    return {'Average_Precision': float(ap), 'Positive_Prevalence': float(prevalence), 'precision': precision.tolist(), 'recall': recall.tolist()}

def compute_roc_auc(df: pd.DataFrame) -> Dict[str, float]:
    """Compute ROC AUC and curve arrays."""
    if df.empty:
        return {'ROC_AUC': np.nan, 'fpr': [], 'tpr': []}
    y_true = df['true_label'].values
    y_score = df['drone_probability'].astype(float).values
    pos_total = (y_true == 1).sum()
    neg_total = (y_true == 0).sum()
    if pos_total == 0 or neg_total == 0:
        return {'ROC_AUC': np.nan, 'fpr': [], 'tpr': []}
    try:
        auc_val = roc_auc_score(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)
    except Exception:
        return {'ROC_AUC': np.nan, 'fpr': [], 'tpr': []}
    return {'ROC_AUC': float(auc_val), 'fpr': fpr.tolist(), 'tpr': tpr.tolist()}

def plot_pr_curve(precision: List[float], recall: List[float], ap: float, prevalence: float, dataset: str, epoch_key: str, out_dir: str):
    if not precision or not recall:
        return
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'PR (AP={ap:.3f})', color='C0', linewidth=2)
    # Baseline (prevalence)
    if not np.isnan(prevalence):
        plt.hlines(prevalence, 0, 1, colors='red', linestyles='--', linewidth=1, label=f'Baseline={prevalence:.3f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    e_label = epoch_key if epoch_key != 'Unknown' else 'Unknown'
    plt.title(f'Precision-Recall: {dataset} (Epoch {e_label})')
    plt.legend(loc='lower left', fontsize=8)
    plt.grid(alpha=0.3)
    safe_dataset = re.sub(r'[^A-Za-z0-9_.-]+', '_', dataset)
    pr_dir = os.path.join(out_dir, 'PR_Curves')
    os.makedirs(pr_dir, exist_ok=True)
    filename = f'Epoch{e_label}_{safe_dataset}_PRCurve.png'
    plt.tight_layout()
    plt.savefig(os.path.join(pr_dir, filename), dpi=200)
    plt.close()

def plot_roc_curve(fpr: List[float], tpr: List[float], auc_val: float, dataset: str, epoch_key: str, out_dir: str):
    if not fpr or not tpr:
        return
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'ROC (AUC={auc_val:.3f})', color='C1', linewidth=2)
    plt.plot([0,1], [0,1], linestyle='--', color='gray', linewidth=1, label='Chance')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    e_label = epoch_key if epoch_key != 'Unknown' else 'Unknown'
    plt.title(f'ROC Curve: {dataset} (Epoch {e_label})')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(alpha=0.3)
    safe_dataset = re.sub(r'[^A-Za-z0-9_.-]+', '_', dataset)
    roc_dir = os.path.join(out_dir, 'ROC_Curves')
    os.makedirs(roc_dir, exist_ok=True)
    filename = f'Epoch{e_label}_{safe_dataset}_ROCCurve.png'
    plt.tight_layout()
    plt.savefig(os.path.join(roc_dir, filename), dpi=200)
    plt.close()

def aggregate_dataset(df: pd.DataFrame) -> Dict[str, float]:
    drone_df = df[df['true_label'] == 1]
    no_drone_df = df[df['true_label'] == 0]

    drone_stats = summarize_probabilities(drone_df)
    no_drone_stats = summarize_probabilities(no_drone_df)
    overall_stats = summarize_probabilities(df)
    # Compute PR-AUC
    pr_stats = compute_pr_auc(df)
    # Compute ROC-AUC
    roc_stats = compute_roc_auc(df)

    mean_delta = (drone_stats['mean'] - no_drone_stats['mean']
                  if not np.isnan(drone_stats['mean']) and not np.isnan(no_drone_stats['mean']) else np.nan)
    mean_ratio = (drone_stats['mean'] / no_drone_stats['mean']
                  if not np.isnan(drone_stats['mean']) and not np.isnan(no_drone_stats['mean']) and no_drone_stats['mean'] != 0 else np.nan)
    pstd = pooled_std(drone_stats['count'], drone_stats['std'], no_drone_stats['count'], no_drone_stats['std'])
    cohen_d = (mean_delta / pstd) if pstd and not np.isnan(pstd) and pstd != 0 else np.nan

    output = {
        # Drone
        'Drone_Count': drone_stats['count'],
        'Drone_Mean': drone_stats['mean'],
        'Drone_Min': drone_stats['min'],
        'Drone_Max': drone_stats['max'],
        'Drone_Median': drone_stats['median'],
        'Drone_Q1': drone_stats['q1'],
        'Drone_Q3': drone_stats['q3'],
        'Drone_Std': drone_stats['std'],
        'Drone_CI95': drone_stats['ci95'],
        # No-Drone
        'NoDrone_Count': no_drone_stats['count'],
        'NoDrone_Mean': no_drone_stats['mean'],
        'NoDrone_Min': no_drone_stats['min'],
        'NoDrone_Max': no_drone_stats['max'],
        'NoDrone_Median': no_drone_stats['median'],
        'NoDrone_Q1': no_drone_stats['q1'],
        'NoDrone_Q3': no_drone_stats['q3'],
        'NoDrone_Std': no_drone_stats['std'],
        'NoDrone_CI95': no_drone_stats['ci95'],
        # Overall
        'Overall_Count': overall_stats['count'],
        'Overall_Mean': overall_stats['mean'],
        'Overall_Min': overall_stats['min'],
        'Overall_Max': overall_stats['max'],
        'Overall_Median': overall_stats['median'],
        'Overall_Q1': overall_stats['q1'],
        'Overall_Q3': overall_stats['q3'],
        'Overall_Std': overall_stats['std'],
        'Overall_CI95': overall_stats['ci95'],
        # Separation metrics
        'Mean_Delta': mean_delta,
        'Mean_Ratio': mean_ratio,
        'Cohen_d': cohen_d,
        'Average_Precision': pr_stats['Average_Precision'],
        'Positive_Prevalence': pr_stats['Positive_Prevalence'],
        'ROC_AUC': roc_stats['ROC_AUC'],
        '_PR_precision': pr_stats['precision'],  # internal use for plotting
        '_PR_recall': pr_stats['recall'],        # internal use for plotting
        '_ROC_fpr': roc_stats['fpr'],            # internal use for plotting
        '_ROC_tpr': roc_stats['tpr'],            # internal use for plotting
    }
    return output


# ---------------- LaTeX Export ----------------

def latex_escape(s: str) -> str:
    if s is None:
        return ''
    repl = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}', '\\': r'\textbackslash{}'
    }
    out = s
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def format_num(x, prec=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ''
    return f"{x:.{prec}f}"


def generate_epoch_latex(epoch_key: str, df_epoch: pd.DataFrame, output_dir: str) -> str:
    # Choose displayed columns for brevity
    display_cols = [
        'Dataset', 'Drone_Count', 'Drone_Mean', 'Drone_Median', 'Drone_Q1', 'Drone_Q3', 'Drone_Std', 'Drone_CI95',
        'NoDrone_Count', 'NoDrone_Mean', 'NoDrone_Median', 'NoDrone_Q1', 'NoDrone_Q3', 'NoDrone_Std', 'NoDrone_CI95',
        'Average_Precision', 'ROC_AUC', 'Positive_Prevalence', 'Mean_Delta', 'Mean_Ratio', 'Cohen_d'
    ]
    sub = df_epoch[display_cols].copy()

    # Sort by largest Mean_Delta descending
    if 'Mean_Delta' in sub.columns:
        sub = sub.sort_values(by='Mean_Delta', ascending=False)

    lines = []
    lines.append('% Auto-generated epoch statistics table')
    caption_epoch = epoch_key if epoch_key != 'Unknown' else 'Unknown'
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append(f'\\caption{{Drone vs No-Drone probability statistics (Epoch {latex_escape(caption_epoch)})}}')
    label_epoch = caption_epoch if caption_epoch != 'Unknown' else 'Unknown'
    lines.append(f'\\label{{tab:epoch-{label_epoch}-stats}}')
    # Column alignment: l then numeric columns right aligned
    col_spec = 'l' + 'r' * (len(display_cols) - 1)
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append('\\toprule')
    header_names = [
        'Dataset', 'n$_D$', 'Mean$_D$', 'Med$_D$', 'Q1$_D$', 'Q3$_D$', 'Std$_D$', 'CI$_D$',
        'n$_N$', 'Mean$_N$', 'Med$_N$', 'Q1$_N$', 'Q3$_N$', 'Std$_N$', 'CI$_N$',
        'AP', 'ROC', 'Prev', '$\\Delta$Mean', 'Ratio', 'd'
    ]
    lines.append(' '.join([h for h in header_names]) + ' \\')
    lines.append('\\midrule')

    for _, row in sub.iterrows():
        vals = []
        for c in display_cols:
            if c == 'Dataset':
                vals.append(latex_escape(str(row[c])))
            elif c.endswith('_Count'):
                vals.append(str(int(row[c])) if not np.isnan(row[c]) else '')
            else:
                prec = 3 if c not in ('Positive_Prevalence',) else 3
                vals.append(format_num(row[c], prec=prec))
        lines.append(' & '.join(vals) + ' \\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    # Escape % properly (LaTeX needs \% and Python needs \\%)
    lines.append('\\footnotesize CI = 95\\% half-width; AP = average precision; Prev = positive prevalence; $d$ = Cohen\'s d; $\\Delta$Mean = Mean$_D$ - Mean$_N$.')
    lines.append('\\end{table}')

    tex_filename = f'Epoch{epoch_key if epoch_key != "Unknown" else "Unknown"}Stats.tex'
    out_path = os.path.join(output_dir, tex_filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return out_path


# ---------------- Main computation ----------------

def compute_epoch_statistics(root: str, debug: bool=False):
    print(f"Scanning for CSV files under: {root}")
    csv_files = find_csv_files_recursively(root)
    if not csv_files:
        print("No CSV files found.")
        return

    print(f"Found {len(csv_files)} CSV file(s). Filtering by required columns...")

    data_store: Dict[str, Dict[str, List[pd.DataFrame]]] = defaultdict(lambda: defaultdict(list))

    valid_files = 0
    skipped = 0

    for csv_path in csv_files:
        if not csv_has_required_columns(csv_path):
            skipped += 1
            if debug:
                print(f"[DEBUG] Skipping (missing columns): {csv_path}")
            continue
        try:
            df = pd.read_csv(csv_path, usecols=list(REQUIRED_COLUMNS))
        except Exception as e:
            print(f"  Error reading {csv_path}: {e}")
            skipped += 1
            continue
        epoch_num = extract_epoch_number(csv_path, root)
        dataset_name = extract_dataset_name(csv_path)
        if debug:
            print(f"[DEBUG] File: {csv_path}\n        -> Epoch: {epoch_num} | Dataset: {dataset_name} | Rows: {len(df)}")
        data_store[epoch_num][dataset_name].append(df)
        valid_files += 1

    print(f"Valid CSV files with required columns: {valid_files}. Skipped: {skipped}.")

    if not data_store:
        print("No qualifying data aggregated. Exiting.")
        return

    # Determine Desktop output path
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    output_dir = os.path.join(desktop_path, 'EpochStats')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Sort epoch keys; treat numeric properly; 'Unknown' last
    def epoch_sort_key(k: str):
        return (1, float('inf')) if k == 'Unknown' else (0, int(k))

    latex_files = []

    for epoch_key in sorted(data_store.keys(), key=epoch_sort_key):
        datasets = data_store[epoch_key]
        rows = []
        if debug:
            print(f"[DEBUG] Aggregating epoch {epoch_key} with datasets: {list(datasets.keys())}")
        for dataset_name, df_list in sorted(datasets.items()):
            merged = merge_stat_records(df_list)
            if merged.empty:
                if debug:
                    print(f"[DEBUG] Dataset {dataset_name} empty after merge.")
                continue
            stats = aggregate_dataset(merged)
            # Extract curve arrays before removing internal keys
            pr_precision = stats.pop('_PR_precision', [])
            pr_recall = stats.pop('_PR_recall', [])
            roc_fpr = stats.pop('_ROC_fpr', [])
            roc_tpr = stats.pop('_ROC_tpr', [])
            stats_row = {'Dataset': dataset_name}
            stats_row.update(stats)
            rows.append(stats_row)
            # Plot PR curve
            try:
                plot_pr_curve(pr_precision, pr_recall, stats_row['Average_Precision'], stats_row['Positive_Prevalence'], dataset_name, epoch_key, output_dir)
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Failed PR curve {dataset_name} epoch {epoch_key}: {e}")
            # Plot ROC curve
            try:
                plot_roc_curve(roc_fpr, roc_tpr, stats_row['ROC_AUC'], dataset_name, epoch_key, output_dir)
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Failed ROC curve {dataset_name} epoch {epoch_key}: {e}")
        if not rows:
            if debug:
                print(f"[DEBUG] No rows produced for epoch {epoch_key} (all empty?)")
            continue
        df_epoch = pd.DataFrame(rows)
        col_order = [
            'Dataset',
            'Drone_Count', 'Drone_Mean', 'Drone_Min', 'Drone_Max', 'Drone_Median', 'Drone_Q1', 'Drone_Q3', 'Drone_Std', 'Drone_CI95',
            'NoDrone_Count', 'NoDrone_Mean', 'NoDrone_Min', 'NoDrone_Max', 'NoDrone_Median', 'NoDrone_Q1', 'NoDrone_Q3', 'NoDrone_Std', 'NoDrone_CI95',
            'Overall_Count', 'Overall_Mean', 'Overall_Min', 'Overall_Max', 'Overall_Median', 'Overall_Q1', 'Overall_Q3', 'Overall_Std', 'Overall_CI95',
            'Mean_Delta', 'Mean_Ratio', 'Cohen_d', 'Average_Precision', 'ROC_AUC', 'Positive_Prevalence'
        ]
        df_epoch = df_epoch[col_order]

        filename = 'EpochUnknownStats.csv' if epoch_key == 'Unknown' else f'Epoch{epoch_key}Stats.csv'
        out_path = os.path.join(output_dir, filename)
        df_epoch.to_csv(out_path, index=False)
        print(f"Wrote {out_path} (datasets: {len(rows)})")

        tex_path = generate_epoch_latex(epoch_key, df_epoch, output_dir)
        latex_files.append(os.path.basename(tex_path))
        print(f"  LaTeX table: {tex_path}")

    # Master file including all tables
    if latex_files:
        master_path = os.path.join(output_dir, 'EpochStats_All.tex')
        with open(master_path, 'w', encoding='utf-8') as f:
            f.write('% Auto-generated master file including all epoch tables\n')
            f.write('\\documentclass{article}\n')
            f.write('\\usepackage{booktabs}\n')
            f.write('\\usepackage[margin=1in]{geometry}\n')
            f.write('\\begin{document}\n')
            for lf in latex_files:
                f.write(f'% ---- {lf} ----\n')
                f.write(f'\\input{{{lf}}}\n\n')
            f.write('\\end{document}\n')
        print(f"Master LaTeX file: {master_path}\nCompile with: pdflatex EpochStats_All.tex inside {output_dir}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description='Compute per-epoch dataset probability statistics and export CSV + LaTeX tables.')
    # Accept multiple tokens for root to handle unquoted paths with spaces
    parser.add_argument('root', nargs='*', default=['.'], help='Root directory to scan (path may contain spaces)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    # Join all positional parts into a single path in case user did not quote it
    root_path = ' '.join(args.root).strip()
    if not root_path:
        root_path = '.'

    if not os.path.exists(root_path):
        print(f"Provided root directory does not exist: {root_path}")
        sys.exit(1)
    compute_epoch_statistics(root_path, debug=args.debug)

if __name__ == '__main__':
    main()
