import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, det_curve, auc, confusion_matrix
import os
import sys
import seaborn as sns

def generate_histogram(df, csv_filepath):
    prob_0 = df[df['true_label'] == 0]['drone_probability']
    prob_1 = df[df['true_label'] == 1]['drone_probability']
    all_probs = df['drone_probability']

    total_samples = len(df)
    if total_samples < 1000:
        num_bins = 100
    elif total_samples < 10000:
        num_bins = 200
    elif total_samples < 8000:
        num_bins = 500
    else:
        num_bins = 2000

    epsilon = 1e-10
    min_prob = max(all_probs.min(), epsilon)
    max_prob = min(all_probs.max(), 1 - epsilon)
    bins = np.logspace(np.log10(min_prob), np.log10(max_prob), num_bins)

    plt.figure(figsize=(12, 8))
    plt.hist(prob_0, bins=bins, alpha=0.6, label=f'No Drone (n={len(prob_0):,})',
             color='red', density=False)
    plt.hist(prob_1, bins=bins, alpha=0.6, label=f'Drone (n={len(prob_1):,})',
             color='blue', density=False)
    plt.yscale('log')
    plt.xscale('log')

    threshold = float(df['aggregation_threshold'].iloc[0])
    plt.axvline(x=threshold, color='black', linestyle='--', alpha=0.7, linewidth=1.5,
                label='Decision Threshold')

    plt.xlabel('Drone Probability (log scale)', fontsize=14)
    plt.ylabel('Count (log scale)', fontsize=14)

    filename = os.path.basename(csv_filepath)
    plt.title(f'{filename}\nBins: {num_bins} | Total: {total_samples:,} samples', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    x_ticks = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]
    valid_ticks = [tick for tick in x_ticks if min_prob <= tick <= max_prob]
    plt.xticks(valid_ticks)

    from matplotlib.ticker import FormatStrFormatter
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    plt.tight_layout()

    base_name = os.path.splitext(csv_filepath)[0]
    plot_filepath = f"{base_name}_histogram.png"
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Plot saved to: {plot_filepath}")
    print(f"  Drone class: mean={prob_1.mean():.4f}, std={prob_1.std():.4f}")
    print(f"  No Drone class: mean={prob_0.mean():.4f}, std={prob_0.std():.4f}")

def generate_confusion_matrix_plot(df, csv_filepath):
    y_true = df['true_label'].to_numpy()
    y_pred = df['predicted_label'].to_numpy()

    # Compute confusion matrix (2x2 for binary classification)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    # Print numeric summary to console
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / cm.sum()
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(f"      [[TN={tn}, FP={fp}],")
    print(f"       [FN={fn}, TP={tp}]]")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  False positive rate: {fp / (fp + tn + 1e-12):.4f}")
    print(f"  False negative rate: {fn / (fn + tp + 1e-12):.4f}")

    # Plot heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_normalized,
        annot=cm,
        fmt='d',
        cmap='Blues',
        xticklabels=['Pred 0 (No Drone)', 'Pred 1 (Drone)'],
        yticklabels=['True 0 (No Drone)', 'True 1 (Drone)'],
        cbar_kws={'label': 'Normalized by true class'}
    )
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix - {os.path.basename(csv_filepath)}')
    plt.tight_layout()

    base_name = os.path.splitext(csv_filepath)[0]
    plot_filepath = f"{base_name}_confusion_matrix.png"
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix plot saved to: {plot_filepath}")

def generate_roc_plot(df, csv_filepath):
    y_true = df['true_label'].values
    y_score = df['drone_probability'].values

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Chance')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {os.path.basename(csv_filepath)}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    base_name = os.path.splitext(csv_filepath)[0]
    plot_filepath = f"{base_name}_roc.png"
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ROC plot saved to: {plot_filepath}")


def generate_det_plot(df, csv_filepath):
    from scipy.stats import norm

    y_true = df['true_label'].values
    y_score = df['drone_probability'].values

    fpr, fnr, _ = det_curve(y_true, y_score, pos_label=1)

    # Clip to avoid infinite probit values, max at 0.5
    fpr_clipped = np.clip(fpr, 1e-4, 0.5)
    fnr_clipped = np.clip(fnr, 1e-4, 0.5)

    # Convert to probit scale
    fpr_probit = norm.ppf(fpr_clipped)
    fnr_probit = norm.ppf(fnr_clipped)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr_probit, fnr_probit, color='purple', lw=2)

    # Set axis ticks in probability space, displayed on probit scale
    ticks = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    tick_labels = [f'{t:.1%}' for t in ticks]
    tick_probit = norm.ppf(ticks)

    plt.xticks(tick_probit, tick_labels)
    plt.yticks(tick_probit, tick_labels)
    plt.xlim(norm.ppf(1e-4), norm.ppf(0.5))
    plt.ylim(norm.ppf(1e-4), norm.ppf(0.5))

    plt.xlabel('False Positive Rate (probit scale)')
    plt.ylabel('False Negative Rate (probit scale)')
    plt.title(f'DET Curve - {os.path.basename(csv_filepath)}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    base_name = os.path.splitext(csv_filepath)[0]
    plot_filepath = f"{base_name}_det.png"
    plt.savefig(plot_filepath, dpi=1000, bbox_inches='tight')
    plt.close()
    print(f"  DET plot saved to: {plot_filepath}")


def find_csv_files_recursively(directory='.'):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.csv'):
                full_path = os.path.join(root, file)
                csv_files.append(full_path)
    return csv_files


def check_csv_columns(csv_filepath):
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


EPS = 1e-12


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_dcf(y_true, scores, threshold, pi=0.5, c_miss=1.0, c_fa=1.0):
    decisions = scores >= threshold
    target_mask = y_true == 1
    n_target = np.sum(target_mask)
    n_non = len(y_true) - n_target
    pmiss = np.mean(~decisions[target_mask]) if n_target else 0.0
    pfa = np.mean(decisions[~target_mask]) if n_non else 0.0
    return pi * c_miss * pmiss + (1 - pi) * c_fa * pfa, pmiss, pfa


def generate_bayes_error_plot(df, csv_filepath, prior_range=(-4, 4), points=1000):
    y_true = df['true_label'].to_numpy()
    probs = df['drone_probability'].clip(EPS, 1 - EPS).to_numpy()
    llr = np.log(probs / (1 - probs))
    prior_logits = np.linspace(prior_range[0], prior_range[1], points)
    priors = _sigmoid(prior_logits)

    e_vals = []
    emin_vals = []
    defaults = np.minimum(priors, 1 - priors)

    fpr, tpr, _ = roc_curve(y_true, probs, pos_label=1)
    fnr = 1 - tpr

    for pi, logit_pi in zip(priors, prior_logits):
        eta = -logit_pi
        decisions = llr >= eta
        miss_mask = y_true == 1
        pmiss = np.mean(~decisions[miss_mask]) if miss_mask.any() else 0.0
        pfa = np.mean(decisions[~miss_mask]) if (~miss_mask).any() else 0.0
        e_vals.append(pi * pmiss + (1 - pi) * pfa)

        costs = pi * fnr + (1 - pi) * fpr
        emin_vals.append(costs.min())

    norm_e = np.divide(e_vals, defaults, out=np.ones_like(e_vals), where=defaults > 0)
    norm_emin = np.divide(emin_vals, defaults, out=np.ones_like(emin_vals), where=defaults > 0)

    plt.figure(figsize=(10, 6))
    plt.plot(prior_logits, norm_e, label='Normalized Bayes error', color='tab:blue')
    plt.plot(prior_logits, norm_emin, label='Normalized minDCF', color='tab:green', linestyle='--')
    plt.axhline(1.0, color='black', linestyle=':', label='Default system')
    plt.xlabel('logit(˜π)')
    plt.ylabel('E(˜π) / E₀(˜π)')
    plt.title(f'Normalized Bayes Error - {os.path.basename(csv_filepath)}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    base_name = os.path.splitext(csv_filepath)[0]
    plot_path = f"{base_name}_bayes_error.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Bayes error plot saved to: {plot_path}")


def choose_analysis_type():
    choices = {
        'hist': generate_histogram,
        'roc': generate_roc_plot,
        'det': generate_det_plot,
        'bayes': generate_bayes_error_plot,
        'cm': generate_confusion_matrix_plot,  # new
    }
    prompt = ("Select analysis type:\n"
              "  hist  - Probability histogram\n"
              "  roc   - ROC curve\n"
              "  det   - DET curve\n"
              "  bayes - Normalized Bayes error plot\n"
              "  cm    - Confusion matrix\n"
              "Enter choice [hist/roc/det/bayes/cm]: ")
    while True:
        choice = input(prompt).strip().lower()
        if choice in choices:
            return choice, choices[choice]
        print("Invalid choice. Please enter 'hist', 'roc', 'det', 'bayes', or 'cm'.")


def process_csv_files(directory='.', analysis_func=None):
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return

    csv_files = find_csv_files_recursively(directory)
    if not csv_files:
        print(f"No CSV files found in directory: {directory}")
        return

    if analysis_func is None:
        print("No analysis function selected.")
        return

    print(f"Found {len(csv_files)} CSV file(s) in directory tree: {directory}")
    print("Checking files for required columns...")

    processed_count = 0
    skipped_count = 0

    for csv_filepath in csv_files:
        print(f"\nChecking {csv_filepath}...")
        if check_csv_columns(csv_filepath):
            print(f"  ✓ File has required columns - processing...")
            try:
                df = pd.read_csv(csv_filepath)
                analysis_func(df, csv_filepath)
                processed_count += 1
            except ValueError as e:
                print(f"  Error processing {csv_filepath}: {e}")
                skipped_count += 1
            except Exception as e:
                print(f"  Unexpected error for {csv_filepath}: {e}")
                skipped_count += 1
        else:
            print(f"  ✗ File does not have required columns - skipping...")
            skipped_count += 1

    print(f"\n=== Summary ===")
    print(f"Total CSV files found: {len(csv_files)}")
    print(f"Files processed: {processed_count}")
    print(f"Files skipped: {skipped_count}")


if __name__ == "__main__":
    analysis_choice, analysis_func = choose_analysis_type()
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    print(f"\nRunning '{analysis_choice}' analysis...")
    process_csv_files(directory, analysis_func)