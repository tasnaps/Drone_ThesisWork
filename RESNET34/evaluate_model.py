"""
Evaluation script for trained ResNet-34 audio classification model.
Uses 1-second sliding windows to match training scale, then aggregates with chosen metric. In thesis we used noisy-OR.
"""
import warnings
import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from config import DATASETS, SAMPLE_RATE, CALIBRATION_SET
from sklearn.exceptions import UndefinedMetricWarning
import librosa
import numpy as np
import torch.nn as nn
import torch
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score,
    roc_curve, accuracy_score
)

from resnet_34 import (
    ResNetForAudioClassification,
    get_label_mappings,
    preprocess_audio_to_resnet_input,
    device
)

from typing import cast

# Optional safetensors loader (used if available)
try:
    from safetensors.torch import load_file as safetensors_load_file  # type: ignore
except Exception:
    safetensors_load_file = None  # type: ignore

# Suppress specific sklearn warnings for single-class datasets
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message=".*No positive samples.*")
warnings.filterwarnings("ignore", message=".*Only one class present.*")
warnings.filterwarnings("ignore", message=".*No positive class found.*")
LABEL2ID = {"unknown": 0, "yes_drone": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}



@dataclass
class EvaluationConfig:
    """Configuration for ResNet-34 evaluation."""
    model_path: str
    data_dir: str
    output_dir: str
    batch_size: int = 64
    window_duration: float = 1.0  # Fixed 1-second windows to match training
    hop_duration: float = 0.5    # 50% overlap
    threshold: float = 0.5
    sample_rate: int = SAMPLE_RATE
    seed: int = 42

class WindowProcessor:
    """Processes audio files using 1-second sliding windows to match training scale."""

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def load_audio_file(self, file_path: str) -> Tuple[Optional[np.ndarray], float]:
        """Load audio file and return waveform and duration."""
        try:
            waveform, sr = librosa.load(file_path, sr=self.config.sample_rate)
            duration = len(waveform) / sr
            return waveform, duration
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, 0.0

    def extract_windows(self, waveform: np.ndarray, duration: float) -> List[np.ndarray]:
        """Extract 1-second windows with specified hop size."""
        window_samples = int(self.config.window_duration * self.config.sample_rate)
        hop_samples = int(self.config.hop_duration * self.config.sample_rate)

        windows = []
        start = 0

        while start + window_samples <= len(waveform):
            window = waveform[start:start + window_samples]
            windows.append(window)
            start += hop_samples

        # Handle remaining audio if significant (>50% of window)
        if len(waveform) - start > window_samples * 0.5:
            remaining = waveform[start:]
            # Pad to full window length
            if len(remaining) < window_samples:
                pad_length = window_samples - len(remaining)
                remaining = np.concatenate([remaining, np.zeros(pad_length)])
            windows.append(remaining[:window_samples])

        return windows

    def aggregate_predictions(self, predictions: List[float], method: str = "noisy_or") -> float:
        """Aggregate predictions from multiple windows."""
        if not predictions:
            return 0.0

        if method == "mean":
            return float(np.mean(predictions))
        elif method == "max":
            return float(np.max(predictions))
        elif method == "median":
            return float(np.median(predictions))
        elif method == "noisy_or":
            # Noisy-OR: P(file) = 1 - ∏(1 - p_window)
            # Clip probabilities to avoid numerical issues
            predictions_clipped = [max(1e-6, min(1-1e-6, p)) for p in predictions]
            log_prob_not_drone = np.sum([np.log1p(-p) for p in predictions_clipped])
            return float(1.0 - np.exp(log_prob_not_drone))
        elif method == "top_k_noisy_or":
            # Top-k Noisy-OR (k=3) to reduce single hot window domination
            k = min(3, len(predictions))
            top_k_preds = sorted(predictions, reverse=True)[:k]
            return self.aggregate_predictions(top_k_preds, "noisy_or")
        else:
            return float(np.mean(predictions))


class ResnetEvaluator:
    """Evaluator for ResNet models with 1s windows, batching, aggregation, and class-index inference."""
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = device  # from resnet_34
        self.model: Optional[nn.Module] = None
        self.processor = WindowProcessor(config)
        # Default to index 1 (drone), will auto-infer during calibration if needed
        self.drone_class_index: int = 1

    def load_model(self):
        """Load ResNet model from file or directory."""
        self.model = load_model(self.config.model_path)
        self.model.to(self.device)
        self.model.eval()

    def _predict_windows(self, windows: List[np.ndarray]) -> List[float]:
        """Return per-window drone probabilities (softmax on logits)."""
        if not windows:
            return []

        probs: List[float] = []
        batch: List[torch.Tensor] = []
        bs = max(1, int(self.config.batch_size))

        def flush():
            if not batch:
                return
            # Ensure model is available for type checkers and safety
            assert self.model is not None, "Model is not loaded in ResnetEvaluator"
            with torch.no_grad():
                x = torch.stack(batch, dim=0).to(self.device)  # [B, C, H, W]
                model = cast(ResNetForAudioClassification, self.model)
                out = model(pixel_values=x)
                logits = out.logits if hasattr(out, "logits") else out
                p = torch.softmax(logits, dim=1)[:, self.drone_class_index].detach().cpu().numpy()
                probs.extend(p.tolist())
            batch.clear()

        for w in windows:
            x = preprocess_audio_to_resnet_input(
                w, sample_rate=self.config.sample_rate
            )  # expected [C, H, W] float32
            if x.ndim == 3:
                batch.append(x)
            else:
                batch.append(x.squeeze(0))
            if len(batch) >= bs:
                flush()
        flush()
        return probs

    def evaluate_file(self, file_path: str, true_label: int, agg: str = "noisy_or") -> Optional[Dict]:
        """Evaluate a single file by windowing and aggregating."""
        waveform, duration = self.processor.load_audio_file(file_path)
        if waveform is None:
            return None

        windows = self.processor.extract_windows(waveform, duration)
        window_probs = self._predict_windows(windows)
        aggregated_prob = self.processor.aggregate_predictions(window_probs, method="noisy_or" if agg is None else agg)
        predicted_label = 1 if aggregated_prob > self.config.threshold else 0

        return {
            "file_path": file_path,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "drone_probability": float(aggregated_prob),
            "aggregation_method": agg or "noisy_or",
            "aggregation_threshold": float(self.config.threshold),
            "split": "windows",
            "duration": float(duration),
            "num_windows": len(windows)
        }

    def evaluate_dataset(self, dataset_name: str, data_dir: str) -> List[Dict]:
        """Evaluate all audio files in a dataset directory."""
        results: List[Dict] = []
        file_count = 0
        dataset_cfg = DATASETS.get(dataset_name, {})
        label_override = dataset_cfg.get("label_override")
        for root, _, files in os.walk(data_dir):
            for f in files:
                if not f.lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
                    continue
                fp = os.path.join(root, f)
                label_str = get_label_for_file(fp, label_override)
                true_label = LABEL2ID.get(label_str, 0)
                r = self.evaluate_file(fp, true_label)
                if r:
                    r["file_id"] = file_count
                    results.append(r)
                    file_count += 1
        return results

    def infer_drone_class_index(self, calibration_dir: str, max_files: int = 100) -> int:
        """Infer whether class index 0 or 1 is the 'drone' class using a balanced sample."""
        if not os.path.isdir(calibration_dir):
            print("Calibration dir not found; keeping default drone_class_index=1")
            return self.drone_class_index

        sample: List[Tuple[str, int]] = []
        d_count = u_count = 0
        for root, _, files in os.walk(calibration_dir):
            for f in files:
                if not f.lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
                    continue
                fp = os.path.join(root, f)
                ls = get_label_for_file(fp, None)
                lab = LABEL2ID.get(ls, 0)
                if lab == 1 and d_count < max_files // 2:
                    sample.append((fp, lab)); d_count += 1
                elif lab == 0 and u_count < max_files // 2:
                    sample.append((fp, lab)); u_count += 1
                if len(sample) >= max_files:
                    break
            if len(sample) >= max_files:
                break

        if len(sample) < 10:
            print("Not enough files to infer class index; keeping default drone_class_index=1")
            return self.drone_class_index

        def eval_idx(idx: int) -> Tuple[float, float]:
            saved = self.drone_class_index
            self.drone_class_index = idx
            d_list, u_list = [], []
            for fp, lab in sample:
                r = self.evaluate_file(fp, lab)
                if not r:
                    continue
                p = float(r["drone_probability"])
                (d_list if lab == 1 else u_list).append(p)
            self.drone_class_index = saved
            d_mean = float(np.mean(d_list)) if d_list else 0.0
            u_mean = float(np.mean(u_list)) if u_list else 0.0
            return d_mean, u_mean

        d0, u0 = eval_idx(0)
        d1, u1 = eval_idx(1)
        sep0, sep1 = d0 - u0, d1 - u1
        chosen = 1 if sep1 >= sep0 else 0
        self.drone_class_index = chosen
        print(f"Inferred drone class index: {chosen} (sep0={sep0:.4f}, sep1={sep1:.4f})")
        return chosen

def print_dataset_results(dataset_name: str, results: List[Dict]):
    """Print evaluation results for a dataset with safe metric calculation."""
    if not results:
        print(f"No results for dataset {dataset_name}")
        return

    # Calculate metrics safely
    true_labels = [r["true_label"] for r in results]
    predicted_labels = [r["predicted_label"] for r in results]
    probabilities = [r["drone_probability"] for r in results]

    accuracy, precision, recall, f1, auc = calculate_metrics_safe(
        true_labels, predicted_labels, probabilities
    )

    # Confusion matrix and other stats
    cm = confusion_matrix(true_labels, predicted_labels)
    class_distribution = Counter(true_labels)
    threshold_used = results[0].get("aggregation_threshold", 0.5)

    # Print results with proper NaN handling
    print(f"Total files processed: {len(results)}")
    print(f"Threshold used: {threshold_used:.6f}")
    print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print(f"Confusion Matrix:")
    print(cm)

def get_label_for_file(file_path: str, label_override: Optional[str]) -> str:
    """Get the appropriate label for a file."""
    if label_override:
        return label_override

    # Extract from folder structure
    parent = Path(file_path).parent.name.lower()

    # Direct mapping
    if parent in LABEL2ID:
        return parent

    # Pattern matching
    drone_patterns = ['drone', 'yes_drone', 'positive', 'uav', 'quadcopter', 'target', '1']
    non_drone_patterns = ['unknown', 'non_drone', 'negative', 'background', 'noise', '0']

    for pattern in drone_patterns:
        if pattern in parent:
            return 'yes_drone'

    for pattern in non_drone_patterns:
        if pattern in parent:
            return 'unknown'

    print(f"Warning: Could not determine label for '{parent}', defaulting to 'unknown'")
    return 'unknown'

def load_model(model_path: str) -> ResNetForAudioClassification:
    """Load trained model from checkpoint."""
    print(f"Loading model from {model_path}")

    # Initialize model architecture
    label2id, _ = get_label_mappings()
    model = ResNetForAudioClassification(num_labels=len(label2id), freeze_blocks=True)

    # Try different checkpoint formats
    checkpoint_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "best_model.pth",
        "final_model.pth"
    ]

    model_loaded = False
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(model_path, checkpoint_file)
        if os.path.exists(checkpoint_path):
            try:
                if checkpoint_file.endswith('.bin') or checkpoint_file.endswith('.pth'):
                    state_dict = torch.load(checkpoint_path, map_location=device)
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    model.load_state_dict(state_dict)
                    print(f"✓ Loaded model from {checkpoint_file}")
                    model_loaded = True
                    break
                elif checkpoint_file.endswith('.safetensors'):
                    if safetensors_load_file is None:
                        print(f"safetensors not available, skipping {checkpoint_file}")
                        continue
                    try:
                        state_dict = safetensors_load_file(checkpoint_path, device=str(device))
                        model.load_state_dict(state_dict)
                        print(f"✓ Loaded model from {checkpoint_file}")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"Failed to load safetensors checkpoint: {e}")
                        continue
            except Exception as e:
                print(f"Failed to load {checkpoint_file}: {e}")
                continue

    if not model_loaded:
        raise FileNotFoundError(f"No valid model checkpoint found in {model_path}")

    model.to(device)
    model.eval()
    return model

def calculate_metrics_safe(true_labels, predicted_labels, probabilities):
    """Calculate metrics with proper handling for single-class datasets."""
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    probabilities = np.array(probabilities)

    # Check if we have both classes
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(predicted_labels)

    # Basic accuracy always works
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Initialize metrics with safe defaults
    precision = recall = f1 = auc = 0.0

    # Only calculate precision/recall/f1 if we have predictions of positive class
    if len(unique_pred) > 1 or 1 in unique_pred:
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average="binary", zero_division=0
        )

    # Only calculate AUC if we have both classes in true labels
    if len(unique_true) > 1:
        try:
            auc = roc_auc_score(true_labels, probabilities)
        except ValueError:
            auc = float('nan')
    else:
        auc = float('nan')

    return accuracy, precision, recall, f1, auc

def calibrate_threshold(model: nn.Module, config: EvaluationConfig, calibration_data_dir: str = None,
                        strategy: str = "f1", force_drone_index: Optional[int] = None) -> Tuple[float, 'ResnetEvaluator']:
    """Robust threshold calibration with improved label inference and tie-breaking."""
    print("\n=== Threshold Calibration ===")
    print(f"Using optimization strategy: {strategy}")

    data_dir = calibration_data_dir if calibration_data_dir else config.data_dir
    print(f"Using calibration dataset: {data_dir}")

    evaluator = ResnetEvaluator(config)
    evaluator.model = model  # model already on device and in eval mode

    # NEW: infer which class index corresponds to 'drone' using the calibration set
    if force_drone_index is not None:
        evaluator.drone_class_index = force_drone_index
        print(f"Using forced drone class index: {force_drone_index}")
    else:
        try:
            evaluator.infer_drone_class_index(data_dir, max_files=100)
        except Exception as e:
            print(f"Warning: Failed to infer class index ({e}); continuing with index {evaluator.drone_class_index}")

    all_probs: List[float] = []
    all_labels: List[int] = []

    # Collect aggregated per-file probabilities
    for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                continue
            file_path = os.path.join(root, file)
            # Determine label from folder structure (no override for calibration set)
            label_string = get_label_for_file(file_path, None)
            true_label = LABEL2ID.get(label_string, 0)

            result = evaluator.evaluate_file(file_path, true_label)
            if result is not None:
                all_probs.append(float(result["drone_probability"]))
                all_labels.append(true_label)

    if len(all_probs) < 10:
        print("Warning: Too few files for proper calibration, using default 0.5")
        return 0.5, evaluator

    labels = np.asarray(all_labels, dtype=np.int32)
    probs = np.asarray(all_probs, dtype=np.float64)

    # Per-class diagnostics
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos, n_neg = int(np.sum(pos_mask)), int(np.sum(neg_mask))

    if n_pos == 0 or n_neg == 0:
        print(
            f"Warning: Only one class present in calibration data (unknown: {n_neg}, drone: {n_pos}), using default 0.5")
        return 0.5, evaluator

    prevalence = float(labels.mean())

    def safe_stats(x: np.ndarray):
        if x.size == 0: return float("nan"), float("nan"), float("nan"), float("nan")
        return float(np.min(x)), float(np.max(x)), float(np.mean(x)), float(np.std(x))

    p_min, p_max, p_mean, p_std = safe_stats(probs)
    n_min, n_max, n_mean, n_std = safe_stats(probs[neg_mask])
    d_min, d_max, d_mean, d_std = safe_stats(probs[pos_mask])
    n_unique = int(np.unique(np.round(probs, 8)).size)

    print(f"Calibration used {len(probs)} files")
    print(f"Class counts -> unknown: {n_neg}, drone: {n_pos}, prevalence: {prevalence:.4f}")
    print(
        f"All probs -> min: {p_min:.8f}, max: {p_max:.8f}, mean: {p_mean:.8f}, std: {p_std:.8f}, unique(~1e-8): {n_unique}")
    print(f"Unknown probs -> min: {n_min:.8f}, max: {n_max:.8f}, mean: {n_mean:.8f}, std: {n_std:.8f}")
    print(f"Drone probs   -> min: {d_min:.8f}, max: {d_max:.8f}, mean: {d_mean:.8f}, std: {d_std:.8f}")

    # If drone mean is still lower and we haven't forced an index yet, try the other index
    if d_mean < n_mean and force_drone_index is None:
        print(f"Switching drone class index to 0 due to inverted probabilities (d_mean={d_mean:.4f} < n_mean={n_mean:.4f})")
        # Use the opposite index (0 if we tried 1, 1 if we tried 0)
        corrected_index = 0 if evaluator.drone_class_index == 1 else 1
        return calibrate_threshold(model, config, calibration_data_dir, strategy, force_drone_index=corrected_index)

    if not np.isfinite(p_std) or p_std < 1e-8 or n_unique <= 2:
        print("Warning: Degenerate probability distribution; returning 0.5")
        return 0.5, evaluator

    # Candidate thresholds
    grid = np.linspace(1e-6, 1 - 1e-6, 1001)
    quantiles = np.quantile(probs, np.linspace(0.01, 0.99, 99))
    candidates = np.unique(np.clip(np.concatenate([grid, quantiles]), 1e-6, 1 - 1e-6))

    # Metrics per threshold
    f1_scores, ba_scores, youden_scores, pos_rates = [], [], [], []

    for thr in candidates:
        preds = (probs >= thr).astype(np.int32)
        tp = np.sum((labels == 1) & (preds == 1))
        tn = np.sum((labels == 0) & (preds == 0))
        fp = np.sum((labels == 0) & (preds == 1))
        fn = np.sum((labels == 1) & (preds == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        f1_scores.append((2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0)
        ba_scores.append(0.5 * (recall + specificity))
        youden_scores.append(recall + specificity - 1.0)
        pos_rates.append(np.mean(preds))
    #Removed balanced acc since same as youden. And we dont need the interpretability.
    metrics_map = {"f1": np.array(f1_scores, dtype=np.float64),
                   "youden": np.array(youden_scores, dtype=np.float64)}
    metric = metrics_map.get(strategy, metrics_map["f1"])
    if strategy not in metrics_map:
        print(f"Warning: Unknown strategy '{strategy}', falling back to f1")

    # Replace non-finite with -inf so they won't win
    metric = np.where(np.isfinite(metric), metric, -np.inf)
    if not np.isfinite(metric).any() or np.all(metric == -np.inf):
        print("Warning: Metric evaluation failed; returning default 0.5")
        return 0.5, evaluator

    best_score = float(np.max(metric))
    best_indices = np.where(np.isclose(metric, best_score, atol=1e-9))[0]

    # Tie-breaking toward prevalence, then median threshold
    best_thresholds = candidates[best_indices]
    best_pos_rates = np.array(pos_rates)[best_indices]

    min_rate_diff = float(np.min(np.abs(best_pos_rates - prevalence)))
    tied_indices = np.where(np.isclose(np.abs(best_pos_rates - prevalence), min_rate_diff, atol=1e-9))[0]

    final_candidates = best_thresholds[tied_indices]
    best_threshold = float(np.median(final_candidates))

    if not (0.0 < best_threshold < 1.0) or not np.isfinite(best_threshold):
        print(f"Warning: Invalid threshold {best_threshold}, using default 0.5")
        return 0.5, evaluator

    print(f"Best threshold ({strategy}): {best_threshold:.6f} (score: {best_score:.4f})")

    # Save threshold with diagnostics
    threshold_file = os.path.join(config.model_path, "ResNet_calibrated_threshold_chosen.json")
    threshold_data = {
        "threshold": best_threshold, "strategy": strategy, "score": best_score,
        "calibration_files": len(probs),
        "class_counts": {"unknown": n_neg, "drone": n_pos, "prevalence": prevalence},
        "unknown_stats": {"min": n_min, "max": n_max, "mean": n_mean, "std": n_std},
        "drone_stats": {"min": d_min, "max": d_max, "mean": d_mean, "std": d_std},
        "all_stats": {"min": p_min, "max": p_max, "mean": p_mean, "std": p_std, "unique": n_unique},
        "calibration_date": datetime.now().isoformat(),
        "drone_class_index": evaluator.drone_class_index
    }

    try:
        with open(threshold_file, 'w') as f:
            json.dump(threshold_data, f, indent=2)
        print(f"Threshold saved to: {threshold_file}")
    except Exception as e:
        print(f"Warning: Failed to save threshold file: {e}")



    return best_threshold, evaluator

# --- Multi-model discovery helpers ---
CHECKPOINT_CANDIDATES = [
    "pytorch_model.bin",
    "model.safetensors",
    "best_model.pth",
    "final_model.pth",
]

def discover_model_dirs(base_path: str) -> List[Tuple[str, str]]:
    """Return a list of (model_name, model_dir) discovered under base_path.
    If base_path itself looks like a model dir (contains a known checkpoint), return only that.
    """
    pairs: List[Tuple[str, str]] = []
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Model path is not a directory: {base_path}")

    # If base contains a checkpoint, treat it as a single model dir
    if any(os.path.exists(os.path.join(base_path, f)) for f in CHECKPOINT_CANDIDATES):
        return [(Path(base_path).name, base_path)]

    # Otherwise, find subdirs that contain a checkpoint
    for name in os.listdir(base_path):
        p = os.path.join(base_path, name)
        if not os.path.isdir(p):
            continue
        if any(os.path.exists(os.path.join(p, f)) for f in CHECKPOINT_CANDIDATES):
            pairs.append((name, p))
    if not pairs:
        raise FileNotFoundError(f"No model checkpoints found in {base_path} or its subdirectories.")
    return pairs


def save_results_to_csv(results: Dict, output_path: str, dataset_name: str = None):
    """Save evaluation results to CSV file in the specified format."""
    # Create CSV data in the exact format requested
    csv_data = []
    for r in results["results"]:
        # Determine split type based on number of windows/chunks
        num_windows = r.get("num_windows") if isinstance(r.get("num_windows"), int) else r.get("num_chunks", 0)
        split = "whole_file" if num_windows == 1 else f"{num_windows}_windows"

        csv_row = {
            "file_id": r.get("file_id"),
            "true_label": r.get("true_label"),
            "predicted_label": r.get("predicted_label"),
            "drone_probability": f"{float(r.get('drone_probability', 0.0)):.8f}",
            "aggregation_method": r.get("aggregation_method", "noisy_or"),
            "aggregation_threshold": f"{float(r.get('aggregation_threshold', results['metrics']['threshold_used'])):.6f}",
            "split": split
        }
        csv_data.append(csv_row)

    # Generate dataset-specific filename if dataset name is provided
    if dataset_name:
        # Extract directory and base filename
        output_dir = os.path.dirname(output_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_output_path = os.path.join(output_dir, f"{dataset_name}_results_{timestamp}.csv")
    else:
        dataset_output_path = output_path

    df = pd.DataFrame(csv_data)
    df.to_csv(dataset_output_path, index=False)
    print(f"Results saved to: {dataset_output_path}")

    # Save summary metrics with dataset-specific filename
    metrics_path = dataset_output_path.replace('.csv', '_metrics.txt')
    with open(metrics_path, 'w') as f:
        metrics = results["metrics"]
        f.write(f"Dataset: {metrics['dataset_name']}\n")
        f.write(f"Total files: {metrics['total_files']}\n")
        f.write(f"Threshold used: {metrics['threshold_used']:.8f}\n")  # Force decimal notation
        f.write(f"Accuracy: {metrics['accuracy']:.8f}\n")  # Force decimal notation
        f.write(f"Precision: {metrics['precision']:.8f}\n")  # Force decimal notation
        f.write(f"Recall: {metrics['recall']:.8f}\n")  # Force decimal notation
        f.write(f"F1-Score: {metrics['f1']:.8f}\n")  # Force decimal notation
        f.write(f"AUC: {metrics['auc']:.8f}\n")  # Force decimal notation
        f.write(f"Class distribution: {metrics['class_distribution']}\n")
        f.write(f"Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}\n")

    print(f"Metrics saved to: {metrics_path}")
    return dataset_output_path

def main():
    """Main evaluation script. Supports single or multiple models within a directory, with optional calibration."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ResNet-34 audio classification model(s)")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to a trained model directory or a directory containing multiple model subdirectories")
    parser.add_argument("--dataset", type=str, default=None,
                      choices=list(DATASETS.keys()),
                      help="Dataset name to evaluate (if not provided, evaluates all datasets)")
    parser.add_argument("--output_dir", type=str,
                      default=None,
                      help="Output directory for results (defaults to timestamped folder on desktop)")
    parser.add_argument("--batch_size", type=int, default=64,
                      help="Batch size for window processing")
    parser.add_argument("--hop_duration", type=float, default=0.5,
                      help="Hop duration between windows in seconds")
    parser.add_argument("--calibrate", action="store_true",
                      help="Calibrate threshold using CALIBRATION_SET before evaluation")
    parser.add_argument("--threshold", type=float, default=None,
                      help="Manual threshold override (skips calibration and saved threshold)")

    parser.add_argument("--threshold_strategy", type=str, default="f1", choices=["f1", "youden"],
                        help="Default strategy choosing")

    args = parser.parse_args()

    # Create timestamped output directory on desktop if not specified
    if args.output_dir is None:
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.dataset:
            args.output_dir = os.path.join(desktop_path, f"resnet34_evaluation_{args.dataset}_{timestamp}")
        else:
            args.output_dir = os.path.join(desktop_path, f"resnet34_evaluation_all_datasets_{timestamp}")

    # Determine which datasets to evaluate
    if args.dataset:
        datasets_to_evaluate = [args.dataset]
        print(f"Evaluating single dataset: {args.dataset}")
    else:
        datasets_to_evaluate = list(DATASETS.keys())
        print(f"Evaluating all {len(datasets_to_evaluate)} datasets")

    # Discover one or more models from the provided path
    try:
        model_pairs = discover_model_dirs(args.model_path)
    except Exception as e:
        print(f"✗ Failed to discover model(s): {e}")
        return

    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved to: {args.output_dir}")

    # Iterate models
    for idx, (model_name, model_dir) in enumerate(model_pairs, start=1):
        print(f"\n{'='*80}")
        print(f"EVALUATING MODEL: {model_name} ({idx}/{len(model_pairs)})")
        print(f"Directory: {model_dir}")
        print(f"{'='*80}")

        # Per-model output dir
        model_out_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_out_dir, exist_ok=True)

        # Load model
        try:
            model = load_model(model_dir)
            print(f"✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load model '{model_name}': {e}")
            continue

        # Base config for this model; per-dataset we update data_dir
        base_config = EvaluationConfig(
            model_path=model_dir,
            data_dir="",
            output_dir=model_out_dir,
            batch_size=args.batch_size,
            hop_duration=args.hop_duration,
            threshold=0.5
        )

        evaluator = ResnetEvaluator(base_config)
        evaluator.model = model

        # Determine threshold: priority manual > saved > calibration > default
        threshold = args.threshold
        threshold_json = os.path.join(model_dir, "ResNet_calibrated_threshold_chosen.json")
        if threshold is None and os.path.exists(threshold_json):
            try:
                with open(threshold_json, 'r') as f:
                    td = json.load(f)
                threshold = float(td.get("threshold", 0.5))
                evaluator.drone_class_index = int(td.get("drone_class_index", evaluator.drone_class_index))
                print(f"✓ Using saved calibrated threshold: {threshold:.6f} (drone_class_index={evaluator.drone_class_index})")
            except Exception as e:
                print(f"Warning: Could not load saved threshold from {threshold_json}: {e}")

        if threshold is None and args.calibrate:
            # Calibrate using configured calibration set
            calib_path = CALIBRATION_SET if isinstance(CALIBRATION_SET, str) else None
            if calib_path and os.path.exists(calib_path):
                thr, calibrated_eval = calibrate_threshold(
                    model,
                    base_config,
                    calib_path,
                    strategy=args.threshold_strategy,
                    force_drone_index=None)
                threshold = thr
                evaluator.drone_class_index = calibrated_eval.drone_class_index
                print(f"✓ Calibrated threshold: {threshold:.6f}; drone_class_index={evaluator.drone_class_index}")
            else:
                print(f"Warning: CALIBRATION_SET path invalid or not set: {CALIBRATION_SET}; skipping calibration")

        if threshold is None:
            threshold = 0.5
            print("Using default threshold: 0.5")

        evaluator.config.threshold = threshold

        # Evaluate each dataset
        for dataset_name in datasets_to_evaluate:
            dataset_config = DATASETS[dataset_name]
            ds_path = dataset_config.get("path")
            if not ds_path or not os.path.exists(ds_path):
                print(f"Skipping {dataset_name}: Dataset path does not exist: {ds_path}")
                continue

            print(f"\n{'-'*60}")
            print(f"DATASET: {dataset_name} ({datasets_to_evaluate.index(dataset_name) + 1}/{len(datasets_to_evaluate)})")
            print(f"Path: {ds_path}")
            print(f"Description: {dataset_config.get('description', 'No description')}")
            if dataset_config.get('label_override'):
                print(f"Label override: {dataset_config['label_override']}")
            print(f"Window duration: {evaluator.config.window_duration}s | Hop: {evaluator.config.hop_duration}s | Threshold: {evaluator.config.threshold:.6f}")

            evaluator.config.data_dir = ds_path
            results_list = evaluator.evaluate_dataset(dataset_name, ds_path)
            print_dataset_results(dataset_name, results_list)

            total_files = len(results_list)
            if total_files == 0:
                print("No files processed; skipping")
                continue

            # Compute metrics
            y_true = np.array([r["true_label"] for r in results_list])
            y_pred = np.array([r["predicted_label"] for r in results_list])
            y_score = np.array([float(r["drone_probability"]) for r in results_list])

            accuracy = float(np.mean(y_true == y_pred))
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )

            try:
                auc = roc_auc_score(y_true, y_score)
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
            except ValueError:
                auc = 0.0
                fpr = tpr = roc_thresholds = None

            try:
                prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
            except ValueError:
                prec_curve = rec_curve = pr_thresholds = None

            cm = confusion_matrix(y_true, y_pred)
            class_distribution = dict(Counter(y_true.tolist()))

            results_dict = {
                "metrics": {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "auc": float(auc),
                    "confusion_matrix": cm.tolist(),
                    "class_distribution": class_distribution,
                    "total_files": total_files,
                    "dataset_name": dataset_name,
                    "threshold_used": float(evaluator.config.threshold),
                },
                "results": results_list,
                "roc_data": {
                    "fpr": fpr.tolist() if fpr is not None else None,
                    "tpr": tpr.tolist() if tpr is not None else None,
                    "thresholds": roc_thresholds.tolist() if roc_thresholds is not None else None
                },
                "pr_data": {
                    "precision": prec_curve.tolist() if prec_curve is not None else None,
                    "recall": rec_curve.tolist() if rec_curve is not None else None,
                    "thresholds": pr_thresholds.tolist() if pr_thresholds is not None else None
                }
            }

            # Save results for this dataset
            dataset_output_dir = os.path.join(model_out_dir, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(dataset_output_dir, f"{dataset_name}_results_{timestamp}.csv")
            save_results_to_csv(results_dict, output_file, dataset_name)

            # Print summary
            m = results_dict["metrics"]
            print(f"Total files processed: {m['total_files']}")
            print(f"Threshold used: {m['threshold_used']:.6f}")
            print(f"Accuracy: {m['accuracy']:.4f} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | F1: {m['f1']:.4f} | AUC: {m['auc']:.4f}")
            print("Confusion Matrix:")
            print(np.array(m['confusion_matrix']))

    print(f"\n✓ Evaluation completed successfully!")
    print(f"All results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
