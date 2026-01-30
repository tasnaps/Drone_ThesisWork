#!/usr/bin/env python3
import os
import soundfile as sf
import numpy as np
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Set

# Map of dataset names to root folders
COMPARISON_DATASETS: Dict[str, str] = {
    "Wonjun Yi": "V:/Wonjun",
    "Al-Emadi": "V:/Al-Emadi/Binary_Drone_Audio",
    "H-2": "V:/H-2/converted/",
    "Drone_print": "V:/DronePrint/DronePrint/Dataset/DS1/ExperimentallyCollected",
    "Authors Collection": "V:/MerilainenCompiledSounds",
    "S&E": "V:/Svanström & Englund/Drone-detection-dataset/Data/Audio",
    "Emo Soundscapes Mixes": "V:/EmoSoundscapes/Parsed",
    "ESC-50": "V:/ESC-50-master/audio",
    "UrbanSound8K": "V:/UrbanSound8K/mergedFolder",
    "FusionDataset": "V:/FusionDataset",
    "Calibration Dataset": "V:/eval_threshold",
}

# Which extensions to consider
AUDIO_EXTS: Set[str] = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

# Configuration for metrics: key, label for plots, use log scale in boxplot
METRICS_CONFIG: Dict[str, Dict[str, Any]] = {
    "length": {"label": "Duration (s)", "log_boxplot": True, "format": ".3f"},
    "rms": {"label": "RMS Amplitude", "log_boxplot": False, "format": ".6f"},
    "peak": {"label": "Peak Amplitude", "log_boxplot": False, "format": ".6f"},
    "crest_factor": {"label": "Crest Factor", "log_boxplot": False, "format": ".6f"},
    "spec_avg": {"label": "Mean Spectrum Magnitude", "log_boxplot": False, "format": ".6f"},
}


def analyze_file(path: str) -> Optional[Tuple[float, float, float, float]]:
    """Return (length_sec, rms, peak, mean_spectrum) or None on error."""
    try:
        data, sr = sf.read(path)
    except Exception:
        # print(f"Warning: Could not read or process {path}: {e}")
        return None

    if data.ndim == 0 or data.shape[0] == 0:  # Handle empty or scalar data
        return None

    # convert to mono if needed
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if data.shape[0] == 0:  # Check again after potential mono conversion if one channel was empty
        return None

    length = data.shape[0] / sr
    rms_val = float(np.sqrt(np.mean(data ** 2)))
    peak_val = float(np.max(np.abs(data)))

    # Ensure data is not all zeros before FFT
    if np.all(data == 0):
        spec_avg = 0.0
    else:
        spec = np.abs(np.fft.rfft(data))
        spec_avg = float(np.mean(spec))

    return length, rms_val, peak_val, spec_avg


def find_audio_files(root: str) -> List[str]:
    """Walk root and return a list of full paths for audio files."""
    audio_files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in AUDIO_EXTS:
                audio_files.append(os.path.join(dirpath, fn))
    return audio_files


def collect_all_metrics(
        datasets: Dict[str, str],
        output_csv_path: str
) -> Dict[str, Dict[str, List[float]]]:
    """Collects metrics for all audio files and writes them to a CSV."""

    collected_metrics: Dict[str, Dict[str, List[float]]] = {
        key: {} for key in METRICS_CONFIG.keys()
    }

    csv_headers = ["dataset", "filepath"] + list(METRICS_CONFIG.keys())

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

        for name, root in datasets.items():
            print(f"\n=== Processing dataset: {name} ({root}) ===")
            files = find_audio_files(root)
            print(f"Found {len(files)} audio files, analyzing…")

            for key in METRICS_CONFIG.keys():
                collected_metrics[key][name] = []

            if not files:
                print(f"No audio files found for dataset {name}.")
                continue

            for path in tqdm(files, desc=name, unit="file"):
                stats = analyze_file(path)
                if stats is None:
                    continue

                length, rms_val, peak_val, spec_avg = stats
                crest_factor = peak_val / rms_val if rms_val > 1e-9 else np.nan

                file_data = {
                    "length": length,
                    "rms": rms_val,
                    "peak": peak_val,
                    "spec_avg": spec_avg,
                    "crest_factor": crest_factor,
                }

                row_values = [name, path]
                for key in METRICS_CONFIG.keys():
                    value = file_data.get(key, np.nan)
                    formatter = METRICS_CONFIG[key]["format"]
                    row_values.append(f"{value:{formatter}}" if not np.isnan(value) else "nan")
                writer.writerow(row_values)

                for key in METRICS_CONFIG.keys():
                    collected_metrics[key][name].append(file_data.get(key, np.nan))

    print(f"\nWrote per-file stats to '{output_csv_path}'")
    return collected_metrics


def generate_histograms(
        all_metrics: Dict[str, Dict[str, List[float]]],
        output_dir_base: str
) -> None:
    """Generates and saves histogram plots for each metric, per dataset."""
    print("\nGenerating per-dataset histograms...")

    # Iterate through each dataset first
    for dataset_name_original in COMPARISON_DATASETS.keys():
        # Sanitize dataset name for directory creation
        sanitized_dataset_name = dataset_name_original.replace(" ", "_").replace("&", "and").replace("/", "_").replace(
            ":", "_")
        dataset_specific_output_dir = os.path.join(output_dir_base, "histograms", sanitized_dataset_name)
        os.makedirs(dataset_specific_output_dir, exist_ok=True)
        print(f"  Generating histograms for dataset: {dataset_name_original} in '{dataset_specific_output_dir}'")

        # Then iterate through each metric for the current dataset
        for metric_key, metric_config in METRICS_CONFIG.items():
            metric_label = metric_config["label"]

            # Get the values for the current dataset and current metric
            # all_metrics structure: {metric_key: {dataset_name_original: [values]}}
            if dataset_name_original not in all_metrics.get(metric_key, {}):
                print(
                    f"    Skipping metric {metric_key} for dataset {dataset_name_original} - no data entry for this metric.")
                continue

            values = all_metrics[metric_key][dataset_name_original]
            valid_values = [v for v in values if not np.isnan(v)]

            if not valid_values:
                print(f"    Skipping metric {metric_key} for dataset {dataset_name_original} - no valid data points.")
                continue

            plt.figure(figsize=(10, 5))
            sns.histplot(valid_values, bins=50, stat="density", element="step", fill=False)

            plt.title(f"{metric_label} Distribution for {dataset_name_original}")
            plt.xlabel(metric_label)
            plt.ylabel("Density")
            plt.tight_layout()

            sane_metric_key = metric_key.replace(" ", "_").replace("/", "_")
            plot_filename = f"hist_{sane_metric_key}.png"
            plt.savefig(os.path.join(dataset_specific_output_dir, plot_filename))
            plt.close()

    print("Per-dataset histograms saved.")


def generate_boxplots(
        all_metrics: Dict[str, Dict[str, List[float]]],
        dataset_names: List[str],
        output_dir: str
) -> None:
    """Generates and saves boxplots for each metric."""
    print("\nGenerating boxplots...")
    boxplot_output_dir = os.path.join(output_dir, "boxplots")
    os.makedirs(boxplot_output_dir, exist_ok=True)

    for key, config in METRICS_CONFIG.items():
        label = config["label"]
        use_log_scale = config["log_boxplot"]

        plt.figure(figsize=(14, 8))

        data_for_plot = []
        for name in dataset_names:
            values = all_metrics[key].get(name, [])
            valid_values = [v for v in values if not np.isnan(v)]
            data_for_plot.append(valid_values)

        if not any(d for d in data_for_plot if d):  # Check if any sublist has data
            print(f"Skipping boxplot for {key} as no data is available for any dataset.")
            plt.close()
            continue

        plt.boxplot(data_for_plot, labels=dataset_names, showfliers=False)
        plt.xticks(rotation=45, ha="right", fontsize="small")
        plt.title(f"Boxplot of {label} by Dataset")

        current_ylabel = label  # Changed from current_xlabel to current_ylabel
        if use_log_scale:
            flat_data = [item for sublist in data_for_plot for item in sublist]
            if all(x > 0 for x in flat_data if x is not None and not np.isnan(x)):  # Ensure positive and not NaN
                plt.yscale("log")
                current_ylabel = f"{label} (log scale)"  # Y-axis label reflects log scale
            else:
                print(f"Warning: Cannot use log scale for y-axis of {label} due to non-positive or all-NaN values.")

        plt.ylabel(current_ylabel)
        plt.tight_layout()
        plt.savefig(os.path.join(boxplot_output_dir, f"box_{key}.png"))
        plt.close()
    print("Boxplots saved.")


def generate_summary_csv(
        all_metrics: Dict[str, Dict[str, List[float]]],
        summary_csv_path: str
) -> None:
    """Generates a summary CSV for audio lengths."""
    print("\nGenerating length summary CSV...")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as sf_csv:
        writer = csv.writer(sf_csv)
        writer.writerow(["dataset", "n_files", "mean_sec", "median_sec", "std_sec", "total_duration_min"])

        for dataset_name, lengths in all_metrics["length"].items():
            valid_lengths = [l for l in lengths if not np.isnan(l)]
            if not valid_lengths:
                print(f"No valid length data for {dataset_name} in summary.")
                writer.writerow([dataset_name, 0, "N/A", "N/A", "N/A", "N/A"])
                continue

            arr = np.array(valid_lengths)
            n_files = len(arr)
            mean_sec = arr.mean()
            median_sec = np.median(arr)
            std_sec = arr.std()
            total_duration_sec = arr.sum()
            total_duration_min = total_duration_sec / 60

            writer.writerow([
                dataset_name,
                n_files,
                f"{mean_sec:.3f}",
                f"{median_sec:.3f}",
                f"{std_sec:.3f}",
                f"{total_duration_min:.2f}"
            ])
            print(f"{dataset_name:25s} files: {n_files:5d}  "
                  f"mean: {mean_sec:7.2f}s  median: {median_sec:7.2f}s  "
                  f"std: {std_sec:6.2f}s  total: {total_duration_min:7.2f}min")

    print(f"Saved per-dataset length summary to '{summary_csv_path}'")


def main() -> None:
    """Main function to run the audio analysis."""
    output_dir_base = "audio_analysis_results"
    os.makedirs(output_dir_base, exist_ok=True)

    per_file_csv_path = os.path.join(output_dir_base, "audio_dataset_stats_detailed.csv")
    length_summary_csv_path = os.path.join(output_dir_base, "audio_dataset_length_summary.csv")

    # 1. Collect all metrics and write detailed CSV
    all_dataset_metrics = collect_all_metrics(COMPARISON_DATASETS, per_file_csv_path)

    # 2. Generate and save per-dataset histogram plots
    # The generate_histograms function now creates subdirectories for each dataset's histograms
    generate_histograms(all_dataset_metrics, output_dir_base)

    # 3. Generate and save boxplots (these will be saved in output_dir_base/boxplots)
    dataset_order_for_plots = list(COMPARISON_DATASETS.keys())
    generate_boxplots(all_dataset_metrics, dataset_order_for_plots, output_dir_base)

    # 4. Generate and save summary CSV for lengths
    generate_summary_csv(all_dataset_metrics, length_summary_csv_path)

    print(f"\nAll analysis complete. Results are in '{output_dir_base}'.")
    print(f"Individual histograms are in '{os.path.join(output_dir_base, 'histograms', '<DatasetName>')}'.")
    print(f"Combined boxplots are in '{os.path.join(output_dir_base, 'boxplots')}'.")


if __name__ == "__main__":
    main()
