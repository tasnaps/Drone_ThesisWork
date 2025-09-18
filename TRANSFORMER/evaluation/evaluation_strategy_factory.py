"""
Evaluation strategy factory for transformer model evaluation.
"""

from abc import ABC, abstractmethod
import time
import pathlib
import numpy as np
import torch
import os
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
from TRANSFORMER.config.config import FileEvaluationConfig, EvaluationConfig

def process_predictions(raw_logits, threshold=0.006):
    """
    Process raw model predictions by applying threshold and extracting probabilities.

    Args:
        raw_logits: Raw logits from model prediction
        threshold: Classification threshold for binary prediction

    Returns:
        tuple: (predictions, probabilities) where predictions are binary and probabilities are drone probabilities
    """
    import torch
    import numpy as np

    # Convert to numpy if needed
    if torch.is_tensor(raw_logits):
        raw_logits = raw_logits.cpu().numpy()

    print(f"DEBUG: Raw logits shape: {raw_logits.shape}")
    print(f"DEBUG: Raw logits sample: {raw_logits[:3] if len(raw_logits) > 0 else 'empty'}")

    # Handle different logit shapes
    if raw_logits.ndim == 2 and raw_logits.shape[1] == 2:
        # Two-class logits: [no_drone, drone]
        drone_logits = raw_logits[:, 1]
        print(f"DEBUG: Using two-class logits (drone column)")
    elif raw_logits.ndim == 2 and raw_logits.shape[1] == 1:
        # Single-class logits
        drone_logits = raw_logits[:, 0]
        print(f"DEBUG: Using single-class logits")
    elif raw_logits.ndim == 1:
        # Already flattened
        drone_logits = raw_logits
        print(f"DEBUG: Using flattened logits")
    else:
        print(f"    ❌ ERROR: Unexpected logits shape: {raw_logits.shape}")
        raise ValueError(f"Unexpected logits shape: {raw_logits.shape}")

    print(f"DEBUG: Drone logits sample: {drone_logits[:5] if len(drone_logits) > 0 else 'empty'}")

    # Apply sigmoid to convert logits to probabilities
    drone_probabilities = 1 / (1 + np.exp(-drone_logits))

    print(f"DEBUG: Probability sample: {drone_probabilities[:5] if len(drone_probabilities) > 0 else 'empty'}")

    # Apply threshold to get binary predictions
    predictions = (drone_probabilities >= threshold).astype(int)

    print(f"Prediction stats: {np.sum(predictions)}/{len(predictions)} positive predictions")
    print(f"Probability range: {drone_probabilities.min():.6f} - {drone_probabilities.max():.6f}")
    print(f"Threshold used: {threshold}")

    # Additional debugging for zero predictions
    if np.sum(predictions) == 0:
        print(f"    ⚠️  WARNING: NO POSITIVE PREDICTIONS!")
        print(f"Max probability: {drone_probabilities.max():.6f}")
        print(f"Probabilities >= 0.001: {np.sum(drone_probabilities >= 0.001)}")
        print(f"Probabilities >= 0.01: {np.sum(drone_probabilities >= 0.01)}")
        print(f"Top 5 probabilities: {np.sort(drone_probabilities)[-5:]}")

    return predictions, drone_probabilities


class EvaluationStrategy(ABC):
    """Abstract base class for evaluation strategies."""

    def __init__(self, model_path, batch_size, output_dir="C:/Users/tapio/PycharmProjects/gradu/eval_results"):
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
        # Validate all datasets before processing
        print("\n" + "="*60)
        print("DATASET VALIDATION AND LABEL ANALYSIS")
        print("="*60)
        
        # Check for non-interactive mode (set by sequential training script)
        non_interactive = os.environ.get('DRONE_EVALUATION_NON_INTERACTIVE', '0') == '1'

        if non_interactive:
            # In non-interactive mode, default to evaluating all datasets
            print("Running in non-interactive mode - evaluating all datasets")
            choice = "N"  # Default to all datasets
        else:
            print("Evaluate only on Fusion dataset? Y/N")
            choice = input()

        if choice == "Y" or choice == "y":
            results = {}
            #Select only Fusion dataset
            if "Fusion" in ENHANCED_DATASETS:
                config = ENHANCED_DATASETS["Fusion"]
                result = self.evaluate_dataset("Fusion", config)
                if result:
                    results["Fusion"] = result
            return results
        else:
            
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

        # Save CSV results
        save_detailed_csv_results(
            results,
            output_dir=self.output_dir,
            file_suffix=f"detailed_{data_type}s",
            data_type=data_type
        )

        print(f"\n{strategy_name} evaluation complete. All plots saved:")
        print(f"  - {prob_plot}")
        print(f"  - {thresh_plot}")
        print(f"\n📁 CSV Results Location:")
        print(f"  - Output directory: {self.output_dir}")
        print(f"  - CSV files should be in: {self.output_dir}/datasets/")

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

        # Process each split using streamlined approach
        for split_idx, ds in enumerate(dataset_splits):
            predictions, probabilities, labels, file_ids = self.streamlined_evaluate_dataset_split(
                ds, dataset_name, split_idx, len(dataset_splits)
            )

            # Adjust file IDs to ensure uniqueness across splits
            adjusted_file_ids = [fid + split_idx * num_files for fid in file_ids]

            # Combine results from all splits
            all_clip_predictions.extend(predictions)
            all_clip_probabilities.extend(probabilities)
            all_clip_labels.extend(labels)
            all_file_ids.extend(adjusted_file_ids)

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

    def load_data(self, dataset_path, label_override, **kwargs):
        """Load data for whole-file evaluation using optimized batch processing."""
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
            target_batch_size=self.batch_size  # Use the configured batch size
        )

    def process_dataset(self, dataset_splits, num_files, dataset_name):
        """Process dataset splits for whole-file evaluation."""
        from TRANSFORMER.evaluation.evaluation_common import process_predictions, monitor_cuda_memory, estimate_memory_requirements

        # Initialize combined results for this dataset
        all_file_predictions = []
        all_file_labels = []
        all_file_ids = []
        all_drone_probabilities = []

        # Process each split
        for split_idx, ds in enumerate(dataset_splits):
            if len(dataset_splits) > 1:
                print(f"Processing {dataset_name} Part {split_idx + 1}/{len(dataset_splits)} ({len(ds)} files)")

            # Check for very large files and manage memory
            if len(ds) == 1:
                original_length = ds[0]['original_length']
                duration_seconds = original_length / SAMPLE_RATE  # SAMPLE_RATE

                if duration_seconds > self.very_large_threshold:
                    print(f"WARNING: Very large file ({duration_seconds:.1f}s) - using aggressive memory management")
                    safe_cuda_cleanup()
                    monitor_cuda_memory("before processing")

                    estimated_memory_gb = estimate_memory_requirements(original_length, SAMPLE_RATE)
                    print(f"Estimated memory needed: {estimated_memory_gb:.2f}GB")

            self.trainer.eval_dataset = ds

            try:
                # Add debugging output before prediction
                print(f"Starting prediction for split {split_idx + 1} with {len(ds)} files...")

                # Get file-level predictions for this split
                file_results = self.trainer.predict(ds)
                print(f"Prediction completed, processing {len(file_results.predictions)} results...")

                raw_logits = file_results.predictions

                # Process predictions
                print(f"Processing predictions with threshold...")
                file_predictions, drone_probabilities = process_predictions(
                    raw_logits, threshold=self.threshold
                )
                print(f"Predictions processed: {len(file_predictions)} files")

                file_labels = file_results.label_ids
                file_ids = ds["file_ids"]
                adjusted_file_ids = [fid + split_idx * num_files for fid in file_ids]

                print(f"Collecting results for {len(file_predictions)} files...")

                # Combine results from all splits
                all_file_predictions.extend(file_predictions)
                all_file_labels.extend(file_labels)
                all_file_ids.extend(adjusted_file_ids)
                all_drone_probabilities.extend(drone_probabilities)

                print(f"    🧹 Cleaning up memory after split {split_idx + 1}...")
                # Clean up memory after each split
                del file_results, raw_logits
                safe_cuda_cleanup()
                print(f"Split {split_idx + 1} completed successfully")

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    print(f"    OOM error on split {split_idx + 1}: {e}")
                    monitor_cuda_memory("after OOM")
                    safe_cuda_cleanup()
                    print(f"    Skipping this split to continue evaluation...")
                    continue
                else:
                    raise e

            except Exception as e:
                print(f"    Error processing split {split_idx + 1}: {e}")
                safe_cuda_cleanup()
                continue

        if not all_file_predictions:
            print(f"No files were successfully processed for {dataset_name}")
            return None

        print(f"Processed {len(all_file_predictions)} files across {len(dataset_splits)} split(s)")

        # Calculate performance metrics
        file_metrics = calculate_performance_metrics(all_file_labels, all_file_predictions, all_drone_probabilities)
        file_metrics.update({
            'num_files': len(all_file_labels),
            'num_splits': len(dataset_splits),
            'probabilities': all_drone_probabilities,
            'true_labels': all_file_labels,
            'file_predictions': all_file_predictions,
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
