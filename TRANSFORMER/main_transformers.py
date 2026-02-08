import os
import sys
# Ensure TensorFlow is not imported by transformers to reduce noise and overhead
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import random
from pathlib import Path

import datasets

ffmpeg_path = r"C:\ffmpeg\bin"  # Adjust to your actual FFmpeg location
if os.path.exists(ffmpeg_path):
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")
    if sys.version_info >= (3, 8):
        os.add_dll_directory(ffmpeg_path)
import warnings
import argparse
import winsound
from win32con import MB_ICONASTERISK
import numpy as np
import torch
from transformers import TrainingArguments, SchedulerType, EarlyStoppingCallback, TrainerCallback
import gc
from data.data_transformers import load_and_split, prepare_dataset_fast, data_collator
from models.model_transformers import model_init, ModelConfig
from models.trainer import WeightedTrainer, compute_metrics, compute_class_weights
from utils.utils_transformers import generate_detailed_report, plot_training_history
import json
from huggingface_hub import snapshot_download
from transformers import AutoConfig
import shutil
import math
from typing import List
from utils.batch_optimization import create_optimized_file_groups
from torch.utils.data import Sampler
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
def _safe_model_dir_name(model_id: str) -> str:
    return model_id.replace('/', '__')

def _toggle_offline(enabled: bool):
    # Keep previous values to restore exactly
    keys = ['TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE']
    prev = {k: os.environ.get(k) for k in keys}
    if enabled:
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
    else:
        for k in keys:
            if k in os.environ:
                del os.environ[k]
    return prev

def _restore_offline(prev: dict):
    for k, v in prev.items():
        if v is None:
            if k in os.environ:
                del os.environ[k]
        else:
            os.environ[k] = v

def ensure_default_model_downloaded(model_id: str,
                                    mirror_into_project: bool = True,
                                    project_store_dir: str = './models/base') -> Path:
    """
    Guarantees the default HF model is available offline.
    1) Try local cache first (no net).
    2) If missing, temporarily go online to download once.
    3) Optionally mirror a copy into the project under `./models/base/<model_id_sanitized>`.
    Returns the local path of the cached snapshot (or project mirror if created).
    """
    # 1) Try HF cache offline
    try:
        # This throws if not cached locally
        _ = AutoConfig.from_pretrained(model_id, local_files_only=True)
        local_path = Path(snapshot_download(repo_id=model_id, local_files_only=True))
    except Exception:
        # 2) Temporarily go online to fetch
        prev = _toggle_offline(False)
        try:
            local_path = Path(snapshot_download(repo_id=model_id))
        finally:
            _restore_offline(prev)

    # 3) Optional project mirror (for portability/backups)
    if mirror_into_project:
        safe_name = _safe_model_dir_name(model_id)
        project_dst = Path(project_store_dir) / safe_name
        project_dst.parent.mkdir(parents=True, exist_ok=True)
        # Use hardlink-friendly copy to avoid duplicating large files if possible
        if not project_dst.exists():
            shutil.copytree(local_path, project_dst)
        return project_dst

    return local_path

def parse_args():
    """Parse command line arguments for transformer training. These override the config settings"""
    parser = argparse.ArgumentParser(description="Train/eval transformer model for drone detection")
    parser.add_argument("--config", help="Path to experiment config file (optional)")
    parser.add_argument("--data-dir", help="Override data directory")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--disable-early-stopping", action="store_true", help="Disable early stopping callback")

    # Audio length modification arguments
    parser.add_argument("--shorten-audio", action="store_true", help="Enable audio shortening during training")
    parser.add_argument("--min-length-ratio", type=float, default=0.5, help="Minimum length ratio for audio shortening (default: 0.5)")
    parser.add_argument("--max-length-ratio", type=float, default=0.75, help="Maximum length ratio for audio shortening (default: 0.75)")
    parser.add_argument("--no-random-crop", action="store_true", help="Use start cropping instead of random cropping")
    return parser.parse_args()

def clear_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Configuration class for better parameter management
class TrainingConfig:
    def __init__(self):
        # Data settings
        self.data_dir = "C:/Users/XXX/Desktop/Datasets/Al-Emadi/Binary_Drone_Audio"#TODO adjust your data dir
        self.augment_data = False  # Enable audio augmentation for training
        self.disable_early_stopping = True  # Disable early stopping callback
        # Training hyperparameters
        self.seed = 42
        self.num_epochs = 20
        self.train_batch_size = 3
        self.eval_batch_size = 3
        self.gradient_accumulation_steps = 2
        self.learning_rate = 1e-5
        self.warmup_steps = 500
        self.early_stopping_patience = 5

        self.use_lr_finder = False
        self.lr_finder_start_lr = 1e-7
        self.lr_finder_end_lr = 1.0
        self.lr_finder_num_iter = 100
        self.lr_finder_output_dir = "./lr_finder_results"
        self.auto_use_suggested_lr = False

        # Audio shortening configuration (no hard truncation)
        self.shorten_audio = False
        self.min_length_ratio = 0.5
        self.max_length_ratio = 0.75
        self.random_crop = True

        # Batch optimization (seconds budget per pre-batched group)
        self.max_batch_seconds = 20.0  # hard cap for seconds per optimized batch

        # Output settings
        self.output_dir = ModelConfig.OUTPUT_DIR
        self.save_total_limit = 50
        self.logging_steps = 500

        self.eval_strategy = "epoch"
        self.eval_steps = 5

        # Hardware optimization
        self.fp16 = torch.cuda.is_available()
        self.dataloader_num_workers = 4 if torch.cuda.is_available() else 2
        self.dataloader_pin_memory = torch.cuda.is_available()

        # Add validation checks for suspicious training patterns
        self.monitor_overfitting = True
        self.loss_threshold_warning = 0.05  # Warn if loss gets suspiciously low
        self.validation_gap_threshold = 0.1  # Warn if val_loss >> train_loss

def setup_training_environment(config):
    """Setup training environment with proper error handling."""
    # Set seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.empty_cache()
        print(f" CUDA available: {torch.cuda.get_device_name()}")
        print(f" GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("Running on CPU")

    # Create output directories
    output_dirs = ["./plots", "./models", "./reports", "./logs"]
    for dir_path in output_dirs:
        Path(dir_path).mkdir(exist_ok=True)

    # Configure warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    print(f"Training environment setup complete")


class OOMReducerTrainer(WeightedTrainer):
    def __init__(self, *args, train_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._oom_retry_attempted = False
        self.custom_train_sampler = train_sampler
        self._step_counter = 0

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns training DataLoader with custom sampler if provided."""
        if self.custom_train_sampler is None:
            return super().get_train_dataloader()

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_sampler=self.custom_train_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to add memory cleanup after gradient accumulation and skip OOM batches."""
        try:
            loss = super().training_step(model, inputs, num_items_in_batch)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
                print("CUDA OOM in training_step - skipping current batch and clearing cache")
                clear_memory()
                # Return a zero loss tensor on the correct device to skip this batch safely
                device = next(model.parameters()).device
                return torch.tensor(0.0, device=device)
            raise
        self._step_counter += 1

        # Aggressive cleanup every 4 accumulation cycles
        if self._step_counter % (self.args.gradient_accumulation_steps * 4) == 0:
            clear_memory()

        return loss

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval, *args, **kwargs):
        """Mirror Trainer signature changes across versions and add cleanup."""
        result = super()._maybe_log_save_evaluate(
            tr_loss, model, trial, epoch, ignore_keys_for_eval, *args, **kwargs
        )

        if self.control.should_evaluate or self.control.should_save:
            clear_memory()

        return result

    def evaluate(self, eval_dataset=None, **kwargs):
        """Override to add memory cleanup after validation."""
        result = super().evaluate(eval_dataset, **kwargs)
        clear_memory()
        return result

    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and not self._oom_retry_attempted:
                self._oom_retry_attempted = True
                print("CUDA OOM detected - retrying with higher accumulation")

                clear_memory()

                # With a batch_sampler, per_device_train_batch_size is ignored by DataLoader.
                # Adjust only accumulation to reduce per-step memory pressure.
                old_per = self.args.per_device_train_batch_size
                old_acc = self.args.gradient_accumulation_steps
                self.args.gradient_accumulation_steps = max(1, old_acc * 2)

                print(
                    f"   Note: using batch_sampler → `per_device_train_batch_size={old_per}` does not change the loader")
                print(f"   Adjusted: accumulation {old_acc}→{self.args.gradient_accumulation_steps}")

                clear_memory()
                return super().train(*args, **kwargs)
            raise

    def save_model(self, output_dir=None, _internal_call=False):
        """Override to add memory cleanup after saving."""
        result = super().save_model(output_dir, _internal_call)
        clear_memory()
        return result

    def _save_checkpoint(self, model, trial):
        """Override to add memory cleanup after checkpoint saving.

        This workspace's installed `transformers.Trainer._save_checkpoint` signature is
        `(self, model, trial)`, so we must not pass any extra args.
        """
        result = super()._save_checkpoint(model, trial)
        clear_memory()
        return result


class MemoryMonitorCallback(TrainerCallback):
    def __init__(self, log_every_n_steps=10):
        self.log_every_n_steps = log_every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every_n_steps == 0 and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9

            print(f"Step {state.global_step} - GPU Memory: "
                  f"{allocated:.2f}GB alloc, {reserved:.2f}GB reserved, "
                  f"{max_allocated:.2f}GB peak")

            # Reset peak memory tracking
            torch.cuda.reset_peak_memory_stats()

    def on_epoch_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            print(f"Epoch {int(state.epoch)} completed - Running memory cleanup...")
            clear_memory()

class SelectiveCheckpointCallback(TrainerCallback):
    """Keeps checkpoints for selected epochs and deletes others right after saving."""

    def __init__(self, keep_epochs=None, epoch_tolerance: float = 1e-2, delete_retries: int = 3):
        # Default: keep epochs 1, 5, 10, 15, 20
        self.keep_epochs = set(keep_epochs or {1, 5, 10, 15, 20})
        self.epoch_tolerance = epoch_tolerance
        self.delete_retries = delete_retries

    def _resolve_checkpoint_path(self, args, state, **kwargs):
        """Best-effort resolve of the checkpoint directory created by this save."""
        # Some versions pass the dir/folder in kwargs
        for k in ("checkpoint_dir", "checkpoint_folder", "output_dir"):
            v = kwargs.get(k)
            if v:
                p = Path(v)
                if p.exists():
                    return p

        # transformers typically uses checkpoint-{global_step}
        p = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if p.exists():
            return p

        # Fallback: most recently modified checkpoint-* folder
        out_dir = Path(args.output_dir)
        candidates = [cp for cp in out_dir.glob("checkpoint-*") if cp.is_dir()]
        if not candidates:
            return None
        candidates.sort(key=lambda cp: cp.stat().st_mtime, reverse=True)
        return candidates[0]

    def on_save(self, args, state, control, **kwargs):
        # Only the main process should delete files
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return

        # With save_strategy='epoch', this should be very close to an integer.
        epoch_float = getattr(state, "epoch", None)
        epoch_done = None
        if epoch_float is not None:
            nearest = int(round(epoch_float))
            if abs(epoch_float - nearest) <= self.epoch_tolerance:
                epoch_done = nearest

        checkpoint_path = self._resolve_checkpoint_path(args, state, **kwargs)

        # If we can't confidently map to an epoch or find the folder, don't delete.
        if checkpoint_path is None or epoch_done is None:
            return

        # Keep desired epochs
        if epoch_done in self.keep_epochs:
            print(f"Keeping checkpoint for epoch {epoch_done}: {checkpoint_path}")
            return

        # Don't delete best checkpoint if Trainer tracks it
        best_ckpt = getattr(state, "best_model_checkpoint", None)
        if best_ckpt and Path(best_ckpt).resolve() == checkpoint_path.resolve():
            print(f"Keeping best checkpoint (epoch {epoch_done}): {checkpoint_path}")
            return

        # Delete with retries (Windows can hold locks briefly)
        for attempt in range(1, self.delete_retries + 1):
            try:
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path)
                break
            except Exception:
                if attempt >= self.delete_retries:
                    print(f"Warning: failed to delete checkpoint: {checkpoint_path}")
                else:
                    import time
                    time.sleep(0.5 * attempt)

        print(f"Removed checkpoint at epoch {epoch_done}: {checkpoint_path}")

def create_training_args(config, total_train_samples):
    # Common arguments for all cases
    target_effective_batch_size = getattr(config, "target_effective_batch_size", 32)
    world_size = max(1, torch.cuda.device_count())  # adjust if using distributed training
    per_device = max(1, config.train_batch_size)  # keep user preference as an upper bound
    accumulation = max(1, math.ceil(target_effective_batch_size / (per_device * world_size)))

    common_args = {
        "output_dir": config.output_dir,
        "per_device_train_batch_size": per_device,
        "per_device_eval_batch_size": config.eval_batch_size,
        "gradient_accumulation_steps": accumulation,
        "num_train_epochs": config.num_epochs,
        "fp16": config.fp16,
        "gradient_checkpointing": True,  # reduce memory by recomputing activations
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": config.save_total_limit,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": SchedulerType.LINEAR,
        "warmup_steps": config.warmup_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f2",
        "greater_is_better": True,
        "logging_dir": "./logs",
        "logging_steps": config.logging_steps,
        "report_to": None,
        "dataloader_num_workers": config.dataloader_num_workers,
        "dataloader_pin_memory": config.dataloader_pin_memory,
        "remove_unused_columns": False,
        "max_grad_norm": 2.0,
        "weight_decay": 0.01,
    }

    print(
        f"Training batch settings -> per_device: {per_device}, accumulation: {accumulation}, effective batch: {per_device * world_size * accumulation}")
    return TrainingArguments(**common_args)

def configure_audio_shortening(config):
    if config.shorten_audio:
        from data.data_transformers import update_audio_length_config
        update_audio_length_config(
            shorten_audio=True,
            min_ratio=config.min_length_ratio,
            max_ratio=config.max_length_ratio,
            random_crop=config.random_crop,
        )
        print(f"Audio shortening enabled: {config.min_length_ratio*100:.0f}%-{config.max_length_ratio*100:.0f}% of original length")
        print(f"Cropping mode: {'Random' if config.random_crop else 'From start'}")
    else:
        from data.data_transformers import update_audio_length_config
        update_audio_length_config(shorten_audio=False)
        print(f"Audio shortening disabled - using full audio files")


class PrecomputedBatchSampler(Sampler):
    """
    A PyTorch Sampler that yields pre-computed batches of indices.
    The batches can be shuffled or sorted by size (descending) each epoch.
    """

    def __init__(self, batches_of_indices: List[List[int]], shuffle: bool = True, sort_by_size: bool = False):
        # Filter out empty batches
        self.batches_of_indices = [batch for batch in batches_of_indices if batch]
        self.shuffle = shuffle
        self.sort_by_size = sort_by_size

        if self.shuffle and self.sort_by_size:
            print("Warning: Both shuffle and sort_by_size are True. Sorting will take precedence.")
            self.shuffle = False

        print(f"Sampler initialized with {len(self.batches_of_indices)} non-empty batches")
        if self.batches_of_indices:
            batch_sizes = [len(batch) for batch in self.batches_of_indices]
            print(f"Batch size range: {min(batch_sizes)}-{max(batch_sizes)} samples")

    def __iter__(self):
        batches = list(self.batches_of_indices)

        if self.sort_by_size:
            # Sort descending to process large batches first (catch OOM early)
            batches.sort(key=len, reverse=True)
        elif self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.batches_of_indices)

def load_and_prepare_data(config):
    print(f"Loading data from: {config.data_dir}")
    ds = load_and_split(config.data_dir)
    raw_train = ds["train"]

    label_col = next(name for name, feat in raw_train.features.items()
                     if isinstance(feat, datasets.ClassLabel))
    num_labels = raw_train.features[label_col].num_classes

    # FIRST: Optimize batches on the RAW dataset (before preprocessing removes audio)
    print("🎯 Optimizing batch grouping by audio length...")

    def get_file_size_mb(path):
        return os.path.getsize(path) / (1024 * 1024)

    optimized_groups = create_optimized_file_groups(
        raw_dataset=raw_train,  # Use raw dataset with audio column
        large_file_threshold=45.0,
        very_large_threshold=180.0,
        max_file_length=config.max_batch_seconds,  # use configurable hard cap
        max_file_size_mb=100,
        get_file_size_mb_func=get_file_size_mb,
        target_batch_size=config.train_batch_size
    )

    # Extract indices for sampler (indices reference raw dataset positions)
    all_batches_of_indices = []
    for category in ["regular", "large", "very_large"]:
        if category in optimized_groups:
            for batch in optimized_groups[category]:
                indices = [file_info.file_idx for file_info in batch]  # Fixed: use .file_idx
                if indices:
                    all_batches_of_indices.append(indices)

    # THEN: Preprocess the dataset (indices will still be valid if no filtering occurs)
    processed_train = prepare_dataset_fast(
        raw_train,
        augment=config.augment_data,
        shorten_for_training=True
    )
    # Validate that preprocessing didn't filter samples
    if len(processed_train) != len(raw_train):
        raise ValueError(
            f"Dataset size mismatch after preprocessing! "
            f"Original: {len(raw_train)}, Processed: {len(processed_train)}. "
            f"Batch indices will be incorrect."
        )

    # Create sampler with indices from raw dataset
    train_sampler = PrecomputedBatchSampler(
        all_batches_of_indices,
        shuffle=False,
        sort_by_size=True
    )

    print(f"✅ Created a custom sampler with {len(train_sampler)} optimized batches.")
    print(f"🚀 Large batches will run first to catch OOM issues early")

    # Validation check
    max_index = max(max(batch) for batch in all_batches_of_indices)
    if max_index >= len(processed_train):
        raise ValueError(
            f"Batch sampler contains invalid index {max_index} "
            f"(dataset size: {len(processed_train)})"
        )
    print(f"✅ All batch indices validated (max index: {max_index})")

    # Set the processed datasets
    ds["train"] = processed_train
    ds["validation"] = prepare_dataset_fast(
        ds["validation"],
        augment=False,
        shorten_for_training=False
    )

    return ds, raw_train, num_labels, train_sampler

def run_learning_rate_finder_if_enabled(config, ds, num_labels):
    if not config.use_lr_finder:
        return
    try:
        from evaluation.learning_rate_finder import run_lr_finder_for_transformer
        print("\n Running Learning Rate Finder...")
        temp_model = model_init(num_labels=num_labels)
        try:
            lr_results = run_lr_finder_for_transformer(
                model=temp_model,
                train_dataset=ds["train"],
                data_collator=data_collator,
                compute_metrics_fn=compute_metrics,
                config=config
            )
            if 'error' not in lr_results:
                suggested_lr = lr_results['suggested_lr']
                print("   LR Finder Results:")
                print(f"   Current LR: {config.learning_rate:.2e}")
                print(f"   Suggested LR: {suggested_lr:.2e}")
                print(f"   Min Loss LR: {lr_results['min_loss_lr']:.2e}")
                print(f"   Steepest Gradient LR: {lr_results['steepest_gradient_lr']:.2e}")
                if config.auto_use_suggested_lr:
                    old_lr = config.learning_rate
                    config.learning_rate = suggested_lr
                    print(f"Updated learning rate: {old_lr:.2e} → {config.learning_rate:.2e}")
            else:
                print(f"LR Finder failed: {lr_results['error']}")
        finally:
            del temp_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error during LR finding: {e}")
        print("Continuing with original learning rate...")


def build_trainer(args, config, ds, raw_train, num_labels, train_sampler=None):
    total_train_samples = len(ds["train"])
    training_args = create_training_args(config, total_train_samples)

    # If we pass a pre-batched sampler, make each step consume exactly one optimized batch
    if train_sampler is not None:
        training_args.per_device_train_batch_size = 1  # critical with batch_sampler
        world_size = max(1, torch.cuda.device_count())
        target_effective = getattr(
            config,
            "target_effective_batch_size",
            config.train_batch_size * max(1, config.gradient_accumulation_steps),
        )
        training_args.gradient_accumulation_steps = max(
            1, math.ceil(target_effective / (1 * world_size))
        )

    cw = compute_class_weights(raw_train)
    print(f"Class weights: {cw}")

    callbacks: List[TrainerCallback] = []
    callbacks.append(SelectiveCheckpointCallback())
    callbacks.append(MemoryMonitorCallback(log_every_n_steps=10))
    if args.disable_early_stopping or config.disable_early_stopping:
        pass
    else:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))

    trainer = OOMReducerTrainer(
        model_init=lambda: model_init(num_labels),
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        train_sampler=train_sampler,
        class_weights=cw,
    )
    return trainer, training_args

def evaluate_and_report(trainer, ds):
    val_results = trainer.evaluate(ds["validation"])
    print("\n=== Validation Results ===")
    for key, value in val_results.items():
        print(f"{key}: {value:.4f}")
    detailed_report, cm = generate_detailed_report(trainer, ds["validation"])
    plot_training_history(trainer)
    return val_results, detailed_report

def save_model_and_feature_extractor(trainer, training_args):
    print(f"Saving final model to: {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    from data.data_transformers import feature_extractor as _fe
    _fe.save_pretrained(training_args.output_dir)

def finalize_and_save_results(training_args, val_results, detailed_report):
    all_results = {
        "validation": val_results,
        "detailed_report": detailed_report,
    }
    with open("./reports/final_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    ##TODO for linux we might want to use another library than winsound
    #winsound.MessageBeep(MB_ICONASTERISK)
    print(f"Model saved to: {training_args.output_dir}")
    print("Results saved to: ./reports/final_results.json")

def handle_args():
    # ─── Parse command line arguments ─────────────────────────────────
    args = parse_args()

    # ─── Initialize configuration with CLI overrides ─────────────────
    config = TrainingConfig()

    # Apply CLI argument overrides
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.train_batch_size = args.batch_size
        config.eval_batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.seed:
        config.seed = args.seed

    # Audio shortening CLI overrides
    if args.shorten_audio:
        config.shorten_audio = True
    if args.min_length_ratio != 0.5:  # Only override if not default
        config.min_length_ratio = args.min_length_ratio
    if args.max_length_ratio != 0.75:  # Only override if not default
        config.max_length_ratio = args.max_length_ratio
    if args.no_random_crop:
        config.random_crop = False

    # Load config file if provided (this would override CLI args)
    if args.config:
        from config.config import load_config
        file_config = load_config(args.config)
        # Apply config file settings here
        print(f"📋 Loaded config from: {args.config}")

    return config, args

def main():
    # Handle args and config as before
    config, args = handle_args()

    # 0) Make sure the default model exists locally; download once if needed
    # Try to read the model id from your ModelConfig; fallback to a known id
    model_id = getattr(ModelConfig, 'MODEL_ID', None) \
               or getattr(ModelConfig, 'MODEL_NAME', None) \
               or 'ALM/wav2vec2-large-audioset'

    base_local_path = ensure_default_model_downloaded(
        model_id=model_id,
        mirror_into_project=True,                # keep a project copy under ./models/base/...
        project_store_dir='./models/base'        # change if you prefer another folder
    )

    # Optional: if you want each fine-tune run saved into a new folder automatically
    # and keep the base model intact, derive a time-stamped output dir here.
    print(f"Output dir: {config.output_dir}")

    # Continue as before
    setup_training_environment(config)
    configure_audio_shortening(config)

    trainer = None
    training_args = None
    val_results = None
    detailed_report = None

    try:
        ds, raw_train, num_labels, train_sampler = load_and_prepare_data(config)
        run_learning_rate_finder_if_enabled(config, ds, num_labels)


        trainer, training_args = build_trainer(args, config, ds, raw_train, num_labels, train_sampler)
        print("Starting single training run...")
        trainer.train()
        print("Training finished.")
        val_results, detailed_report = evaluate_and_report(trainer, ds)
        save_model_and_feature_extractor(trainer, training_args)
        finalize_and_save_results(training_args, val_results, detailed_report)

    except Exception as e:
        # unchanged error handling...
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        try:
            if trainer is not None and training_args is not None:
                print("Saving partial model and results...")
                trainer.save_model(training_args.output_dir)
                partial_results = {
                    "validation": val_results,
                }
                with open("./reports/partial_results.json", 'w') as f:
                    json.dump(partial_results, f, indent=2)
                print(f"Model saved to: {training_args.output_dir}")
                print("Partial results saved to: ./reports/partial_results.json")
        except Exception as save_e:
            print(f"Error saving partial results: {save_e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Cleared cache and collected garbage")

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Final cleanup done")

if __name__ == "__main__":
    main()
