"""
Evaluation strategy factory for transformer model evaluation.
"""
from TRANSFORMER.utils.batch_optimization import plan_batches_for_dataset, subset_dataset
from .evaluation_common import calculate_performance_metrics, process_predictions
from abc import ABC, abstractmethod
import time
import pathlib
import numpy as np
import torch
import os
import json
from datetime import datetime
from sklearn.metrics import f1_score
from typing import List, Dict, Any
from .evaluation_common import (
    load_model_and_feature_extractor, create_trainer_config,
    safe_cuda_cleanup, calculate_performance_metrics, print_performance_statistics
)
from .evaluation_utils import (
    save_detailed_csv_results, plot_probability_distributions, plot_threshold_analysis,
    print_performance_analysis, print_evaluation_summary, print_detailed_prediction_summary
)
from TRANSFORMER.data.data_loading import load_and_prepare_clip_based
from TRANSFORMER.config.dataset_config import ENHANCED_DATASETS, validate_dataset_labels, SAMPLE_RATE
from TRANSFORMER.config.config import EvaluationConfig

#def process_predictions(raw_logits, threshold=0.006):
#    """
#    Process raw model predictions by applying threshold and extracting probabilities.
#
#    Args:
#        raw_logits: Raw logits from model prediction
#        threshold: Classification threshold for binary prediction
#
#    Returns:
#        tuple: (predictions, probabilities) where predictions are binary and probabilities are drone probabilities
#    """
#    import torch
#    import numpy as np
#
#    # Convert to numpy if needed
#    if torch.is_tensor(raw_logits):
#        raw_logits = raw_logits.cpu().numpy()
#
#    # Handle different logit shapes
#    if raw_logits.ndim == 2 and raw_logits.shape[1] == 2:
#        # Two-class logits: [no_drone, drone]
#        drone_logits = raw_logits[:, 1]
#    elif raw_logits.ndim == 2 and raw_logits.shape[1] == 1:
#        # Single-class logits
#        drone_logits = raw_logits[:, 0]
#    elif raw_logits.ndim == 1:
#        # Already flattened
#        drone_logits = raw_logits
#    else:
#        raise ValueError(f"Unexpected logits shape: {raw_logits.shape}")
#
#    # Apply sigmoid to convert logits to probabilities
#    drone_probabilities = 1 / (1 + np.exp(-drone_logits))
#
#    # Apply threshold to get binary predictions
#    predictions = (drone_probabilities >= threshold).astype(int)
#
#
#    return predictions, drone_probabilities


class EvaluationStrategy(ABC):
    """Abstract base class for evaluation strategies."""

    def __init__(self, model_path, batch_size, output_dir="C:/Users/XXX/PycharmProjects/gradu/eval_results"):
        self.model_path = model_path
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.model = None
        self.feat_ext = None
        self.device = None
        self.trainer = None

    def setup(self):
        """Initialize model, feature extractor, and trainer."""
        self.model, self.feat_ext, self.device = load_model_and_feature_extractor(self.model_path)
        self.trainer = create_trainer_config(
            self.model, self.feat_ext, SAMPLE_RATE, self.batch_size, self.output_dir
        )

    @abstractmethod
    def load_data(self, dataset_path, label_override, **kwargs):
        """Load and prepare data for evaluation."""
        pass

    @abstractmethod
    def process_dataset(self, dataset_splits, num_files, dataset_name):
        """Process dataset splits and return results."""
        pass

    @abstractmethod
    def get_strategy_name(self):
        """Return the name of this evaluation strategy."""
        pass

    @abstractmethod
    def get_data_type(self):
        """Return the data type ('clip' or 'file')."""
        pass

    def load_saved_threshold(self, threshold_file: str | None) -> bool:
        """Load a previously calibrated threshold from JSON and apply it."""
        try:
            if threshold_file and os.path.exists(threshold_file):
                with open(threshold_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                thr = data.get('threshold')
                if thr is not None:
                    self.threshold = float(thr)
                    print(f"Loaded calibrated threshold from '{threshold_file}': {self.threshold:.6f}")
                    return True
        except Exception as e:
            print(f"Failed to load threshold from '{threshold_file}': {e}")
        return False

    def save_threshold(self, threshold_value: float, threshold_file: str | None, meta: dict | None = None):
        """Persist the calibrated threshold to JSON."""
        if not threshold_file:
            return
        try:
            os.makedirs(os.path.dirname(threshold_file), exist_ok=True)
            payload = {
                "threshold": float(threshold_value),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "strategy": self.get_strategy_name(),
                "data_type": self.get_data_type(),
                "model_path": getattr(self, "model_path", None),
            }
            if meta:
                payload.update(meta)
            with open(threshold_file, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
            print(f"Saved calibrated threshold to '{threshold_file}': {threshold_value:.6f}")
        except Exception as e:
            print(f"Failed to save threshold to '{threshold_file}': {e}")

    # --- UPDATED: full-dataset threshold calibration (no 80/20 split) ---
    def calibrate_threshold(
        self,
        calibration_key: str = "Calibration",
        threshold_file: str | None = None,
        metric: str = "f1",
        calibration_fraction: float = 1.0
    ) -> float:
        """
        Calibrate threshold using the entire calibration dataset:
        - Collect file-level probabilities and labels from the calibration set
        - Pick the threshold that maximizes F1 on the whole set
        - Save and apply the threshold
        """
        if calibration_key not in ENHANCED_DATASETS:
            raise ValueError(f"Calibration key '{calibration_key}' not found in ENHANCED_DATASETS")

        ds_cfg = ENHANCED_DATASETS[calibration_key]
        # Be tolerant to different config field names
        dataset_path = ds_cfg.get("path") or ds_cfg.get("dir") or ds_cfg.get("root")
        label_override = ds_cfg.get("label_override", None)

        print(f"\n=== Calibrating threshold on '{calibration_key}' (fraction={calibration_fraction:.3f}) ===")
        print(f"Current threshold before calibration: {self.threshold:.6f}")

        # Validate calibration_fraction
        try:
            calibration_fraction = float(calibration_fraction)
        except Exception:
            raise ValueError(f"Invalid calibration_fraction: {calibration_fraction}")

        if calibration_fraction <= 0.0:
            raise ValueError("calibration_fraction must be > 0.0")
        if calibration_fraction > 1.0:
            print(f"Warning: calibration_fraction > 1.0 ({calibration_fraction}) - treating as 1.0 (full dataset)")
            calibration_fraction = 1.0

        # Load calibration data using the concrete strategy's loader
        dataset_splits, num_files = self.load_data(
            dataset_path=dataset_path,
            label_override=label_override,
            dataset_name=calibration_key,
            is_calibration=True,  # Add this flag
            test_fraction=float(calibration_fraction)
        )

        print(f"Loaded calibration data: {len(dataset_splits)} splits, {num_files} files")

        # Collect file-level probabilities and labels without applying any threshold
        probs, labels = self._collect_file_probs_and_labels(
            dataset_splits=dataset_splits,
            num_files=num_files,
            dataset_name=calibration_key
        )

        print(f"Collected {len(probs)} probabilities and {len(labels)} labels")
        if len(probs) > 0:
            print(f"Probability range: {np.min(probs):.6f} to {np.max(probs):.6f}")
            print(f"Sample probabilities: {probs[:5] if len(probs) >= 5 else probs}")

        if len(labels) > 0:
            unique_labels = np.unique(labels)
            print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

        probs = np.asarray(probs, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)

        if probs.size == 0 or labels.size == 0:
            raise RuntimeError("No calibration data collected for threshold calibration")

        # If there is no class variance, keep current threshold and warn
        if np.unique(labels).size < 2:
            print("Warning: Calibration set has a single class. Keeping existing threshold.")
            calibrated = float(self.threshold)
            self.save_threshold(calibrated, threshold_file, meta={"calibration_key": calibration_key, "metric": metric})
            print(f"Calibrated threshold (unchanged): {calibrated:.6f}")
            return calibrated

        # Candidate thresholds: all unique prob values plus edges
        unique_probs = np.unique(probs)
        # Guard against degenerate case of all-equal probs
        candidates = np.unique(np.concatenate(([0.0], unique_probs, [1.0])))

        print(f"  Trying {len(candidates)} threshold candidates...")
        print(f"  Probability range: {probs.min():.6f} to {probs.max():.6f}")
        print(f"  Initial threshold: {self.threshold:.6f}")

        best_thr = 0.5  # Default to middle value instead of current threshold
        best_score = -1.0
        best_inverted = False

        # First try normal thresholds (probs > threshold)
        print("  Testing normal thresholds (prob > threshold)...")
        for t in candidates:
            preds = (probs > t).astype(np.int32)
            score = f1_score(labels, preds, average="binary", zero_division=0.0)
            if score > best_score:
                best_score = score
                best_thr = float(t)
                best_inverted = False
                print(f"    New best: threshold={t:.6f}, F1={score:.4f}")

        print(f"  Best normal threshold: {best_thr:.6f} (F1={best_score:.4f})")

        # If we get poor performance, try inverted logic (probs < threshold)
        # This handles cases where the model predicts the opposite of ground truth
        if best_score < 0.5:  # If F1 is poor, try inverting
            print(f"  Low F1 score ({best_score:.4f}) detected, trying inverted thresholds...")
            for t in candidates:
                preds = (probs < t).astype(np.int32)  # Inverted logic
                score = f1_score(labels, preds, average="binary", zero_division=0.0)
                if score > best_score:
                    best_score = score
                    best_thr = -float(t)  # Negative threshold indicates inverted logic
                    best_inverted = True
                    print(f"    New best INVERTED: threshold=-{t:.6f}, F1={score:.4f}")

        # If still no good threshold found, use a reasonable default
        if best_score <= 0.0:
            print("  Warning: No threshold achieved F1 > 0. Using median probability as fallback.")
            median_prob = np.median(probs)
            best_thr = float(median_prob)
            best_inverted = False
            print(f"  Fallback threshold: {best_thr:.6f}")

        self.threshold = best_thr
        threshold_info = {
            "calibration_key": calibration_key,
            "metric": metric,
            "score": float(best_score),
            "num_files": int(num_files),
            "num_records": int(probs.size),
            "inverted_logic": best_inverted
        }
        self.save_threshold(self.threshold, threshold_file, meta=threshold_info)

        pos = int(labels.sum())
        neg = int(labels.size - pos)
        print(f"Calibration set size: {labels.size} (pos={pos}, neg={neg})")

        if best_inverted:
            print(f"Selected INVERTED threshold: {self.threshold:.6f} (F1={best_score:.4f}) - using < logic")
        else:
            print(f"Selected threshold: {self.threshold:.6f} (F1={best_score:.4f}) - using > logic")

        return self.threshold

    # --- Each concrete strategy must implement this for calibration ---
    @abstractmethod
    def _collect_file_probs_and_labels(self, dataset_splits, num_files, dataset_name):
        """
        Return:
            - list/np.array of per-file probabilities (P(yes_drone))
            - list/np.array of per-file true labels (0/1)
        """
        raise NotImplementedError

    def evaluate_dataset(self, dataset_name, config):
        """Evaluate a single dataset using this strategy."""
        print(f"\n{'='*50}")
        print(f"Processing {dataset_name}: {config['description']}")
        print(f"Path: {config['path']}")
        print(f"Strategy: {self.get_strategy_name()}")
        print(f"Label strategy: {'Override: ' + config['label_override'] if config['label_override'] else 'Folder-based detection'}")

        # Verify dataset path exists
        if not pathlib.Path(config['path']).exists():
            print(f"⚠️  Dataset path not found: {config['path']}")
            return None

        start = time.time()

        try:
            # Load data using strategy-specific method
            dataset_splits, num_files = self.load_data(
                config['path'],
                config['label_override']
            )

            if not dataset_splits:
                print(f"⚠️  No audio files found in {dataset_name}")
                return None

            # Process dataset using strategy-specific method
            metrics = self.process_dataset(dataset_splits, num_files, dataset_name)

            if metrics:
                # Print performance statistics
                print_performance_statistics(dataset_name, metrics, metrics)
                print(f"Processing took {time.time() - start:.1f}s")

                safe_cuda_cleanup()
                return metrics
            else:
                return None

        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            safe_cuda_cleanup()
            return None

    def evaluate_all_datasets(self):
        """Evaluate all datasets using this strategy."""
        print(f"🔍 EVALUATE_ALL: Starting with threshold = {self.threshold:.6f}")

        # Validate all datasets before processing
        print("\n" + "="*60)
        print("DATASET VALIDATION AND LABEL ANALYSIS")
        print("="*60)
        
        # Check for non-interactive mode (set by sequential training script)
        non_interactive = os.environ.get('DRONE_EVALUATION_NON_INTERACTIVE', '0') == '1'

        #if non_interactive:
        #    # In non-interactive mode, default to evaluating all datasets
        #    print("Running in non-interactive mode - evaluating all datasets")
        #    choice = "N"  # Default to all datasets
        #else:
        #    print("Evaluate only on Fusion dataset? Y/N")
        #    choice = input()

        #if choice == "Y" or choice == "y":
        #    results = {}
        #    #Select only Fusion dataset
        #    if "Fusion" in ENHANCED_DATASETS:
        #        config = ENHANCED_DATASETS["Fusion"]
        #        result = self.evaluate_dataset("Fusion", config)
        #        if result:
        #            results["Fusion"] = result
        #    return results
        #else:
            
        for dataset_name, config in ENHANCED_DATASETS.items():
            validate_dataset_labels(config['path'], dataset_name, config['label_override'])
        # Evaluate all datasets
        results = {}
        for dataset_name, config in ENHANCED_DATASETS.items():
            result = self.evaluate_dataset(dataset_name, config)
            if result:
                results[dataset_name] = result
        return results

    def generate_outputs(self, results):
        """Generate all output files and plots."""
        print(f"🔍 GENERATE_OUTPUTS: Starting with threshold = {self.threshold:.6f}")

        if not results:
            print("No results to generate outputs for.")
            return

        strategy_name = self.get_strategy_name()
        data_type = self.get_data_type()

        # Print summaries
        print_evaluation_summary(results, script_type=data_type)
        print_detailed_prediction_summary(results)

        # Generate plots
        title_suffix = f" ({strategy_name})"
        save_prefix = f"{data_type.lower()}_"

        prob_plot = plot_probability_distributions(results, title_suffix=title_suffix, save_prefix=save_prefix)
        thresh_plot = plot_threshold_analysis(results, title_suffix=title_suffix, save_prefix=save_prefix)

        # Print performance analysis
        print_performance_analysis(results, data_type=data_type)

        # Save CSV results with the calibrated threshold
        save_detailed_csv_results(
            results,
            output_dir=self.output_dir,
            file_suffix=f"detailed_{data_type}s",
            data_type=data_type,
            threshold=self.threshold  # Pass the calibrated threshold
        )

        print(f"\n{strategy_name} evaluation complete. All plots saved:")
        print(f"  - {prob_plot}")
        print(f"  - {thresh_plot}")
        print(f"\n📁 CSV Results Location:")
        print(f"  - Output directory: {self.output_dir}")
        print(f"  - CSV files should be in: {self.output_dir}/datasets/")
        if self.threshold < 0:
            print(f"  - Using calibrated INVERTED threshold: {self.threshold:.6f} (< logic)")
        else:
            print(f"  - Using calibrated threshold: {self.threshold:.6f}")

        # Check if CSV files were actually created
        csv_dir = os.path.join(self.output_dir, "datasets")
        if os.path.exists(csv_dir):
            csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
            print(f"  - Found {len(csv_files)} CSV files:")
            for csv_file in csv_files[:5]:  # Show first 5 files
                print(f"    • {csv_file}")
            if len(csv_files) > 5:
                print(f"    • ... and {len(csv_files) - 5} more")
        else:
            print(f"  - CSV directory does not exist: {csv_dir}")

        return results


class ClipBasedEvaluationStrategy(EvaluationStrategy):
    """Strategy for clip-based evaluation with streamlined file-level focus."""

    def __init__(self, model_path, clip_duration=None, batch_size=None, max_clips_per_dataset=None,
                 aggregation_threshold=None, output_dir="./eval_results"):
        # Load configuration
        config = EvaluationConfig()

        # Use config values as defaults, allow override via parameters
        self.clip_duration = clip_duration if clip_duration is not None else config.clip.clip_duration
        batch_size = batch_size if batch_size is not None else config.clip.batch_size
        self.max_clips_per_dataset = max_clips_per_dataset if max_clips_per_dataset is not None else config.clip.max_clips_per_dataset
        self.aggregation_threshold = aggregation_threshold if aggregation_threshold is not None else config.clip.aggregation_threshold

        # Store the threshold as an instance variable
        self.threshold = config.clip.threshold

        super().__init__(model_path, batch_size, output_dir)
        self.clip_samples = int(16000 * self.clip_duration)  # Assuming 16kHz sample rate

    def _collect_file_probs_and_labels(self, dataset_splits, num_files, dataset_name):
        """
        Collect file-level probabilities and labels from clip-based data.
        This reuses the normal processing flow to avoid duplicating inference logic.
        """
        # Process all clips in the dataset
        all_clip_predictions = []
        all_clip_probabilities = []
        all_clip_labels = []
        all_file_ids = []

        # Process each split
        for split_idx, ds in enumerate(dataset_splits):
            predictions, probabilities, labels, file_ids = self.streamlined_evaluate_dataset_split(
                ds, dataset_name, split_idx, len(dataset_splits)
            )

            # Adjust file IDs to ensure uniqueness across splits
            adjusted_file_ids = [fid + split_idx * num_files for fid in file_ids]

            all_clip_predictions.extend(predictions)
            all_clip_probabilities.extend(probabilities)
            all_clip_labels.extend(labels)
            all_file_ids.extend(adjusted_file_ids)

        # Aggregate clips to file level using confidence-weighted averaging
        from TRANSFORMER.evaluation.evaluation_common import aggregate_predictions_by_confidence

        file_predictions, file_labels, file_probabilities = aggregate_predictions_by_confidence(
            all_clip_predictions, all_clip_labels, all_file_ids, all_clip_probabilities,
            threshold=self.aggregation_threshold, verbose=False
        )

        return file_probabilities, file_labels

    def get_strategy_name(self):
        return "Clip-based (File-level focused)"

    def get_data_type(self):
        return "clip"

    def load_data(self, dataset_path, label_override, **kwargs):
        """Load data for clip-based evaluation."""
        return load_and_prepare_clip_based(
            dataset_path,
            label_override,
            self.max_clips_per_dataset,
            self.clip_samples
        )

    def streamlined_evaluate_dataset_split(self, dataset, dataset_name, split_idx=0, total_splits=1):
        """Streamlined evaluation focusing on file-level results only."""
        if total_splits > 1:
            print(f"  Processing {dataset_name} Part {split_idx + 1}/{total_splits} ({len(dataset)} clips)")

        # Process dataset in batches
        all_predictions = []
        all_probabilities = []
        all_true_labels = []
        all_file_ids = []

        for i in range(0, len(dataset), self.batch_size):
            batch = dataset.select(range(i, min(i + self.batch_size, len(dataset))))

            with torch.no_grad():
                trainer_output = self.trainer.predict(batch)
                predictions = trainer_output.predictions

                # Use the instance threshold instead of relying on defaults
                batch_preds, batch_probs = process_predictions(predictions, threshold=self.threshold)

                all_predictions.extend(batch_preds)
                all_probabilities.extend(batch_probs)
                all_true_labels.extend(batch['labels'])
                all_file_ids.extend(batch['file_ids'])

        return all_predictions, all_probabilities, all_true_labels, all_file_ids

    def process_dataset(self, dataset_splits, num_files, dataset_name):
        """Process dataset splits for streamlined clip-based evaluation."""
        from TRANSFORMER.evaluation.evaluation_common import aggregate_predictions_by_confidence

        print(f"Evaluating {dataset_name} with streamlined file-level focus...")

        # Initialize combined results for this dataset
        all_clip_predictions = []
        all_clip_probabilities = []
        all_clip_labels = []
        all_file_ids = []

        cumulative_files = 0
        # Process each split using streamlined approach
        for split_idx, ds in enumerate(dataset_splits):
            predictions, probabilities, labels, file_ids = self.streamlined_evaluate_dataset_split(
                ds, dataset_name, split_idx, len(dataset_splits)
            )

            # Adjust file IDs to ensure uniqueness across splits
            adjusted_file_ids = [fid + cumulative_files for fid in file_ids]

            # Combine results from all splits
            all_clip_predictions.extend(predictions)
            all_clip_probabilities.extend(probabilities)
            all_clip_labels.extend(labels)

            all_file_ids.extend(adjusted_file_ids)

            # Update cumulative count by number of UNIQUE files in this split
            cumulative_files += len(set(file_ids))

            # Clean up memory after each split
            del ds
            safe_cuda_cleanup()

        print(f"Processed {len(all_clip_predictions)} clips from {len(set(all_file_ids))} files across {len(dataset_splits)} split(s)")

        # Aggregate to file level using confidence-weighted averaging
        file_predictions, file_labels, file_probabilities = aggregate_predictions_by_confidence(
            all_clip_predictions, all_clip_labels, all_file_ids, all_clip_probabilities,
            threshold=self.aggregation_threshold, verbose=False
        )

        # Calculate file-level metrics only
        file_metrics = calculate_performance_metrics(
            file_labels, file_predictions, file_probabilities
        )

        # Return streamlined results with consistent naming
        return {
            'dataset_name': dataset_name,
            'num_files': len(np.unique(all_file_ids)),
            'num_clips': len(all_clip_predictions),
            'num_splits': len(dataset_splits),
            'predictions': file_predictions,
            'true_labels': file_labels,
            'probabilities': file_probabilities,
            # File-level metrics with prefix for clarity
            **{f'file_level_{k}': v for k, v in file_metrics.items()},
            # Backward compatibility - include metrics without prefix
            **file_metrics,
            # Keep these for compatibility with existing code
            'file_predictions': file_predictions,
            'file_labels': file_labels
        }


class WholeFileEvaluationStrategy(EvaluationStrategy):
    """Strategy for whole-file evaluation."""

    def __init__(self, model_path, batch_size=None, large_file_threshold=None,
                 very_large_threshold=None, max_file_length=None,
                 max_file_size_mb=None, output_dir="./eval_results"):
        # Load configuration
        config = EvaluationConfig()

        # Use config values as defaults, allow override via parameters
        batch_size = batch_size if batch_size is not None else config.file.batch_size
        self.large_file_threshold = large_file_threshold if large_file_threshold is not None else config.file.large_file_threshold
        self.very_large_threshold = very_large_threshold if very_large_threshold is not None else config.file.very_large_threshold
        self.max_file_length = max_file_length if max_file_length is not None else config.file.max_file_length
        self.max_file_size_mb = max_file_size_mb if max_file_size_mb is not None else config.file.max_file_size_mb

        # Store the threshold as an instance variable
        self.threshold = config.file.threshold

        super().__init__(model_path, batch_size, output_dir)

    def get_strategy_name(self):
        return "Whole-file"

    def get_data_type(self):
        return "file"

    def _collect_file_probs_and_labels(self, dataset_splits, num_files, dataset_name):
        """
        Collect raw probabilities and labels for calibration WITHOUT applying any threshold.
        This avoids circular dependency during calibration.
        """
        from TRANSFORMER.evaluation.evaluation_common import monitor_cuda_memory, safe_cuda_cleanup
        from TRANSFORMER.utils.batch_optimization import plan_batches_for_dataset, subset_dataset

        print(f"  Collecting raw probabilities for calibration from {dataset_name}...")

        all_probabilities = []
        all_labels = []

        for split_idx, ds in enumerate(dataset_splits):
            print(f"    Processing calibration split {split_idx + 1}/{len(dataset_splits)} ({len(ds)} files)")

            try:
                batches, lengths, _ = plan_batches_for_dataset(
                    ds,
                    batch_size=self.batch_size,
                    sample_rate=SAMPLE_RATE,
                    large_file_threshold=self.large_file_threshold,
                    very_large_threshold=self.very_large_threshold,
                )
            except Exception as e:
                print(f"    Fallback to single-batch for calibration split {split_idx + 1}: {e}")
                batches = [list(range(len(ds)))]
                lengths = [SAMPLE_RATE] * len(ds)

            for b_i, idxs in enumerate(batches):
                mb = subset_dataset(ds, idxs)

                try:
                    self.trainer.eval_dataset = mb
                    file_results = self.trainer.predict(mb)
                    raw_logits = file_results.predictions

                    # Get RAW probabilities without applying any threshold
                    from scipy.special import softmax
                    probabilities = softmax(raw_logits, axis=1)
                    drone_probabilities = probabilities[:, 1]  # P(yes_drone)

                    file_labels = file_results.label_ids

                    # Extract original file IDs and aggregate by file (same as in process_dataset)
                    try:
                        if "part_metadata" in mb.column_names:
                            part_metadata = mb["part_metadata"]
                            original_file_ids = [metadata["original_file_id"] for metadata in part_metadata]
                        else:
                            original_file_ids = [f"calib_{split_idx}_{b_i}_{i}" for i in range(len(drone_probabilities))]
                    except Exception as e:
                        original_file_ids = [f"calib_error_{split_idx}_{b_i}_{i}" for i in range(len(drone_probabilities))]

                    # Group by file ID and take max probability per file (same aggregation as evaluation)
                    file_groups = {}
                    for prob, label, file_id in zip(drone_probabilities, file_labels, original_file_ids):
                        if file_id not in file_groups:
                            file_groups[file_id] = {'probs': [], 'label': label}
                        file_groups[file_id]['probs'].append(prob)

                    # Take max probability per file (consistent with evaluation logic)
                    for file_id, group in file_groups.items():
                        max_prob = max(group['probs'])
                        all_probabilities.append(max_prob)
                        all_labels.append(group['label'])

                except Exception as e:
                    print(f"    Error in calibration microbatch {b_i + 1}: {e}")
                    continue
                finally:
                    safe_cuda_cleanup()

            safe_cuda_cleanup()

        print(f"  Collected {len(all_probabilities)} file-level probabilities for calibration")
        return all_probabilities, all_labels

    def load_data(self, dataset_path, label_override, **kwargs):
        """Load data for whole-file evaluation using optimized batch processing."""
        # Always use our optimized loader. We no longer try a HuggingFace split/fraction path.
        from TRANSFORMER.data.data_loading import load_and_prepare_whole_file_optimized

        return load_and_prepare_whole_file_optimized(
            dataset_path,
            label_override,
            max_files_per_split=1000,
            large_file_threshold=self.large_file_threshold,
            very_large_threshold=self.very_large_threshold,
            max_file_length=self.max_file_length,
            max_file_size_mb=self.max_file_size_mb,
            trainer=self.trainer,
            target_batch_size=self.batch_size  # Used only as a hint for splitting
        )

        # Fallback to the standard optimized loader for full datasets or non-calibration
        return load_and_prepare_whole_file_optimized(
            dataset_path,
            label_override,
            max_files_per_split=1000,
            large_file_threshold=self.large_file_threshold,
            very_large_threshold=self.very_large_threshold,
            max_file_length=self.max_file_length,
            max_file_size_mb=self.max_file_size_mb,
            trainer=self.trainer,
            target_batch_size=self.batch_size  # Use the configured batch size
        )

    def process_dataset(self, dataset_splits, num_files, dataset_name):
        """Process dataset splits for whole-file evaluation with proper part aggregation."""
        from TRANSFORMER.evaluation.evaluation_common import monitor_cuda_memory, \
            estimate_memory_requirements, safe_cuda_cleanup


        # Use original_file_id as the grouping key instead of file paths
        file_results_dict = {}  # key: original_file_id, value: list of (prediction, probability, label)

        for split_idx, ds in enumerate(dataset_splits):
            if len(dataset_splits) > 1:
                print(f"Processing {dataset_name} Part {split_idx + 1}/{len(dataset_splits)} ({len(ds)} files)")

            try:
                batches, lengths, _ = plan_batches_for_dataset(
                    ds,
                    batch_size=self.batch_size,
                    sample_rate=SAMPLE_RATE,
                    large_file_threshold=self.large_file_threshold,
                    very_large_threshold=self.very_large_threshold,
                )
            except Exception as e:
                print(f"  Fallback to single-batch for split {split_idx + 1}: {e}")
                batches = [list(range(len(ds)))]
                lengths = [SAMPLE_RATE] * len(ds)

            total_len_sec = sum(lengths) / max(1, SAMPLE_RATE)
            print(f"  Split {split_idx + 1}: {len(ds)} files, ~{total_len_sec:.1f}s total, {len(batches)} microbatches")

            for b_i, idxs in enumerate(batches):
                mb = subset_dataset(ds, idxs)
                mb_len_sec = sum(lengths[i] for i in idxs) / max(1, SAMPLE_RATE)

                print(f"    ▶ Predict microbatch {b_i + 1}/{len(batches)} "
                      f"({len(idxs)} files, ~{mb_len_sec:.1f}s audio total)")

                try:
                    self.trainer.eval_dataset = mb
                    file_results = self.trainer.predict(mb)
                    raw_logits = file_results.predictions

                    file_predictions, drone_probabilities = process_predictions(
                        raw_logits, threshold=self.threshold
                    )

                    file_labels = file_results.label_ids

                    # Extract original file IDs from part_metadata
                    try:
                        if "part_metadata" in mb.column_names:
                            part_metadata = mb["part_metadata"]
                            original_file_ids = [metadata["original_file_id"] for metadata in part_metadata]
                        else:
                            # Fallback to generated IDs
                            original_file_ids = [f"generated_{split_idx}_{b_i}_{i}" for i in
                                                 range(len(file_predictions))]
                    except Exception as e:
                        print(f"    Error extracting original_file_ids: {e}")
                        original_file_ids = [f"error_{split_idx}_{b_i}_{i}" for i in range(len(file_predictions))]

                    # Group results by original file ID
                    for pred, prob, label, orig_file_id in zip(file_predictions, drone_probabilities, file_labels,
                                                               original_file_ids):
                        if orig_file_id not in file_results_dict:
                            file_results_dict[orig_file_id] = []
                        file_results_dict[orig_file_id].append((pred, prob, label))

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "out of memory" in str(e).lower():
                        print(f"    OOM in microbatch {b_i + 1}: {e}")
                        monitor_cuda_memory("after OOM")
                        safe_cuda_cleanup()
                        continue
                    else:
                        raise
                except Exception as e:
                    print(f"    Error in microbatch {b_i + 1}: {e}")
                    safe_cuda_cleanup()
                    continue
                finally:
                    try:
                        del file_results, raw_logits
                    except Exception:
                        pass
                    safe_cuda_cleanup()

            del ds
            safe_cuda_cleanup()

        # Show aggregation summary
        multi_part_files = {fid: len(results) for fid, results in file_results_dict.items() if len(results) > 1}
        if multi_part_files:
            print(f"Aggregating {len(multi_part_files)} multi-part files into single file results")

        if not file_results_dict:
            print(f"No files were successfully processed for {dataset_name}")
            return None

        # Aggregate results - combine parts of the same original file
        all_file_predictions = []
        all_file_labels = []
        all_file_ids = []
        all_drone_probabilities = []

        for file_id, (original_file_id, results) in enumerate(file_results_dict.items()):
            if not results:
                continue

            preds, probs, labels = zip(*results)

            # For multi-part files: aggregate using max probability and prediction
            final_prediction = max(preds)
            final_probability = max(probs)
            final_label = labels[0]  # All parts should have the same label

            # Verify all labels are consistent for multi-part files
            if len(set(labels)) > 1:
                print(f"Warning: Inconsistent labels for file {original_file_id}: {set(labels)}")

            all_file_predictions.append(final_prediction)
            all_drone_probabilities.append(final_probability)
            all_file_labels.append(final_label)
            all_file_ids.append(file_id)


        file_metrics = calculate_performance_metrics(all_file_labels, all_file_predictions, all_drone_probabilities)
        file_metrics.update({
            'num_files': len(all_file_labels),
            'num_splits': len(dataset_splits),
            'probabilities': all_drone_probabilities,
            'true_labels': all_file_labels,
            'file_predictions': all_file_predictions,
            'file_ids': all_file_ids,
            'file_labels': all_file_labels
        })

        return file_metrics


class EvaluationStrategyFactory:
    """Factory for creating evaluation strategies."""

    @staticmethod
    def create_strategy(strategy_type, model_path, **kwargs):
        """
        Create an evaluation strategy.

        Args:
            strategy_type: Type of strategy ("clip" or "file")
            model_path: Path to the model
            **kwargs: Additional arguments for the strategy

        Returns:
            EvaluationStrategy instance
        """
        if strategy_type.lower() in ["clip", "clip-based", "clips"]:
            return ClipBasedEvaluationStrategy(model_path, **kwargs)
        elif strategy_type.lower() in ["file", "whole-file", "files", "whole_file"]:
            return WholeFileEvaluationStrategy(model_path, **kwargs)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}. Use 'clip' or 'file'.")

    @staticmethod
    def get_available_strategies():
        """Return list of available strategy types."""
        return ["clip", "file"]


def run_evaluation(strategy_type, model_path, **kwargs):
    """
    Convenience function to run evaluation with a specific strategy.

    Args:
        strategy_type: Type of strategy ("clip" or "file")
        model_path: Path to the model
        **kwargs: Additional arguments for the strategy

    Returns:
        Dictionary of results by dataset
    """
    # Create strategy
    strategy = EvaluationStrategyFactory.create_strategy(strategy_type, model_path, **kwargs)

    # Setup model and trainer
    strategy.setup()

    # Evaluate all datasets
    results = strategy.evaluate_all_datasets()

    # Generate outputs
    strategy.generate_outputs(results)

    return results


class SequentialEvaluationRunner:
    """Runner for sequential evaluation of multiple models or configurations."""

    def __init__(self, base_config: dict[str, any]):
        self.base_config = base_config
        self.results = {}
        self.failed_runs = []
    #TODO Check usage
    def run_sequential_evaluations(self, evaluation_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run multiple evaluations sequentially.

        Args:
            evaluation_configs: List of configuration dictionaries, each containing
                               overrides for the base configuration

        Returns:
            Dictionary mapping run names to their results
        """
        print(f"\nStarting Sequential Evaluation Runner")
        print(f"📋 {len(evaluation_configs)} configurations to process")
        print("="*80)

        for i, config_override in enumerate(evaluation_configs, 1):
            # Merge base config with override
            merged_config = self._merge_configs(self.base_config, config_override)
            run_name = config_override.get('run_name', f"run_{i}")

            print(f"\n Processing {run_name} ({i}/{len(evaluation_configs)})")
            print(f"Model: {merged_config.get('model_path', 'default')}")
            print(f"Strategy: {merged_config.get('strategy_type', 'file')}")

            try:
                # Run single evaluation
                start_time = time.time()
                results = self._run_single_evaluation(merged_config)

                if results:
                    elapsed = time.time() - start_time
                    self.results[run_name] = {
                        'results': results,
                        'config': merged_config,
                        'elapsed_time': elapsed,
                        'status': 'success'
                    }
                    print(f"✅ {run_name} completed successfully in {elapsed:.1f}s")
                else:
                    self.failed_runs.append({
                        'run_name': run_name,
                        'config': merged_config,
                        'error': 'No results returned'
                    })
                    print(f"{run_name} failed: No results returned")

            except Exception as e:
                self.failed_runs.append({
                    'run_name': run_name,
                    'config': merged_config,
                    'error': str(e)
                })
                print(f"{run_name} failed: {e}")

            # Cleanup between runs
            safe_cuda_cleanup()

        # Generate summary report
        self._generate_summary_report()

        return self.results

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        merged = base.copy()

        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    def _run_single_evaluation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single evaluation with the given configuration."""
        strategy_type = config.get('strategy_type', 'file')
        model_path = config.get('model_path')
        output_dir = config.get('output_dir', './eval_results')

        # Extract strategy-specific parameters
        if strategy_type == "clip":
            strategy_params = {
                'clip_duration': config.get('clip_duration'),
                'batch_size': config.get('clip_batch_size'),
                'max_clips_per_dataset': config.get('max_clips'),
                'aggregation_threshold': config.get('clip_threshold')
            }
        else:  # file
            strategy_params = {
                'batch_size': config.get('file_batch_size'),
                'large_file_threshold': config.get('large_file_threshold'),
                'very_large_threshold': config.get('very_large_threshold'),
                'max_file_length': config.get('max_file_length'),
                'max_file_size_mb': config.get('max_file_size_mb')
            }

        # Filter out None values
        strategy_params = {k: v for k, v in strategy_params.items() if v is not None}

        # Run evaluation using existing function
        return run_evaluation(
            strategy_type=strategy_type,
            model_path=model_path,
            output_dir=output_dir,
            **strategy_params
        )

    def _generate_summary_report(self):
        """Generate and save a summary report of all runs."""
        print(f"\n📊 Sequential Evaluation Summary")
        print("="*80)
        print(f"Successful runs: {len(self.results)}")
        print(f"❌ Failed runs: {len(self.failed_runs)}")

        if self.results:
            print(f"\n Successful Evaluations:")
            for run_name, data in self.results.items():
                elapsed = data['elapsed_time']
                num_datasets = len(data['results'])
                print(f"  📈 {run_name}: {num_datasets} datasets in {elapsed:.1f}s")

        if self.failed_runs:
            print(f"\nFailed Evaluations:")
            for failed in self.failed_runs:
                print(f"  {failed['run_name']}: {failed['error']}")

        # Save detailed report
        report_path = "sequential_evaluation_report.json"
        try:
            import json
            with open(report_path, 'w') as f:
                json.dump({
                    'successful_runs': self.results,
                    'failed_runs': self.failed_runs,
                    'summary': {
                        'total_runs': len(self.results) + len(self.failed_runs),
                        'successful_count': len(self.results),
                        'failed_count': len(self.failed_runs)
                    }
                }, f, indent=2, default=str)
            print(f"\n Detailed report saved to: {report_path}")
        except Exception as e:
            print(f"⚠Could not save report: {e}")


def run_sequential_evaluation(base_config: Dict[str, Any],
                            evaluation_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convenience function to run sequential evaluations.

    Args:
        base_config: Base configuration that will be used for all runs
        evaluation_configs: List of configuration overrides for each run

    Returns:
        Dictionary of results from all runs
    """
    runner = SequentialEvaluationRunner(base_config)
    return runner.run_sequential_evaluations(evaluation_configs)
