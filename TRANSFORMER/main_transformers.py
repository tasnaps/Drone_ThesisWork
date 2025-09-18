import random
from pathlib import Path
import os
import warnings
import argparse
import winsound
from win32con import MB_ICONASTERISK
import numpy as np
import torch
from transformers import TrainingArguments, SchedulerType, EarlyStoppingCallback
import gc
from data.data_transformers import load_and_split, prepare_dataset_fast, data_collator
from models.model_transformers import model_init, ModelConfig
from models.trainer import WeightedTrainer, compute_metrics, compute_class_weights
from utils.utils_transformers import generate_detailed_report, plot_training_history
import json
import shutil


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

# Configuration class for better parameter management
class TrainingConfig:
    def __init__(self):
        # Data settings
        self.data_dir = "C:/Gradu Juttui/Datasets/C/"
        self.augment_data = False  # Enable audio augmentation for training
        self.disable_early_stopping = True  # Disable early stopping callback
        # Training hyperparameters
        self.seed = 42
        self.num_epochs = 0.1
        self.train_batch_size = 16
        self.eval_batch_size = 16
        self.gradient_accumulation_steps = 2
        self.learning_rate = 1e-5
        self.warmup_steps = 500
        self.early_stopping_patience = 3

        self.use_lr_finder = False
        self.lr_finder_start_lr = 1e-7
        self.lr_finder_end_lr = 1.0
        self.lr_finder_num_iter = 100
        self.lr_finder_output_dir = "./lr_finder_results"
        self.auto_use_suggested_lr = False

        # Audio shortening configuration
        self.shorten_audio = False
        self.min_length_ratio = 0.5
        self.max_length_ratio = 0.75
        self.random_crop = True

        # Output settings
        self.output_dir = ModelConfig.OUTPUT_DIR
        self.save_total_limit = 5
        self.logging_steps = 250

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

def create_training_args(config):
    """Create optimized training arguments."""
    return TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        fp16=config.fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=config.save_total_limit,
        learning_rate=config.learning_rate,
        lr_scheduler_type=SchedulerType.LINEAR,
        warmup_steps=config.warmup_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir="./logs",
        logging_steps=config.logging_steps,
        report_to=None,
        # Memory and performance optimizations
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        remove_unused_columns=False,  # Keep all columns for audio data
        # Additional stability improvements
        max_grad_norm=2.0,  # Increased from 1.0 to handle current gradient patterns
        weight_decay=0.01,  # L2 regularization
    )

def configure_audio_shortening(config):
    if config.shorten_audio:
        from data.data_transformers import update_audio_length_config
        update_audio_length_config(
            shorten_audio=True,
            min_ratio=config.min_length_ratio,
            max_ratio=config.max_length_ratio,
            random_crop=config.random_crop
        )
        print(f"Audio shortening enabled: {config.min_length_ratio*100:.0f}%-{config.max_length_ratio*100:.0f}% of original length")
        print(f"Cropping mode: {'Random' if config.random_crop else 'From start'}")
    else:
        print(f"Audio shortening disabled - using full audio files")

def load_and_prepare_data(config):
    print(f"Loading data from: {config.data_dir}")
    ds = load_and_split(config.data_dir)
    raw_train = ds["train"]
    print(f"Dataset sizes - Train: {len(ds['train'])}, Val: {len(ds['validation'])}, Test: {len(ds['test'])}")
    # Detect label column
    from datasets import ClassLabel as _CL
    try:
        label_col = next(name for name, feat in raw_train.features.items() if isinstance(feat, _CL))
        num_labels = raw_train.features[label_col].num_classes
        print(f"Found label column: {label_col} with {num_labels} classes")
    except StopIteration:
        raise ValueError("No ClassLabel feature found in the dataset")
    # Preprocess
    print("Fast preprocessing - processing all audio files upfront...")
    ds["train"] = prepare_dataset_fast(ds["train"], augment=config.augment_data, shorten_for_training=False)
    ds["validation"] = prepare_dataset_fast(ds["validation"], augment=False, shorten_for_training=False)
    ds["test"] = prepare_dataset_fast(ds["test"], augment=False, shorten_for_training=False)
    print("All audio preprocessing complete - data is now cached and ready for fast training!")
    return ds, raw_train, num_labels

def run_learning_rate_finder_if_enabled(config, ds, num_labels):
    if not config.use_lr_finder:
        return
    try:
        from evaluation.learning_rate_finder import run_lr_finder_for_transformer
        print("\n🔍 Running Learning Rate Finder...")
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

def build_trainer(args, config, ds, raw_train, num_labels):
    training_args = create_training_args(config)
    cw = compute_class_weights(raw_train)
    print(f"Class weights: {cw}")
    callbacks = []
    if args.disable_early_stopping or config.disable_early_stopping:
        print(f"Either Args or Config early stopping: {config.disable_early_stopping} disabled training will run for full {config.num_epochs} epochs)")
    else:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))
        print(f"Early stopping enabled with patience: {config.early_stopping_patience}")
    trainer = WeightedTrainer(
        class_weights=cw,
        model=model_init(num_labels=num_labels),
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    return trainer, training_args

def evaluate_and_report(trainer, ds):
    val_results = trainer.evaluate(ds["validation"])
    test_results = trainer.evaluate(ds["test"])
    print("\n=== Validation Results ===")
    for key, value in val_results.items():
        print(f"{key}: {value:.4f}")
    print("\n=== Test Results ===")
    for key, value in test_results.items():
        print(f"{key}: {value:.4f}")
    detailed_report, cm = generate_detailed_report(trainer, ds["test"])
    plot_training_history(trainer)
    return val_results, test_results, detailed_report

def save_model_and_feature_extractor(trainer, training_args):
    print(f"Saving final model to: {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    from data.data_transformers import feature_extractor as _fe
    _fe.save_pretrained(training_args.output_dir)

def finalize_and_save_results(training_args, val_results, test_results, detailed_report):
    all_results = {
        "validation": val_results,
        "test": test_results,
        "detailed_report": detailed_report,
    }
    with open("./reports/final_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    ##TODO for linux we might want to use another library than winsound
    winsound.MessageBeep(MB_ICONASTERISK)
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

    #Handle args for config override
    config, args = handle_args()
    # ─── Setup training environment ───────────────────────────────────
    setup_training_environment(config)

    # ─── Configure audio length shortening based on config ──────────────
    configure_audio_shortening(config)

    # Initialize for exception safety
    trainer = None
    training_args = None
    val_results = None
    test_results = None

    try:
        ds, raw_train, num_labels = load_and_prepare_data(config)
        run_learning_rate_finder_if_enabled(config, ds, num_labels)
        trainer, training_args = build_trainer(args, config, ds, raw_train, num_labels)
        print("Starting single training run...")
        trainer.train()
        print("Training finished.")
        val_results, test_results, detailed_report = evaluate_and_report(trainer, ds)
        save_model_and_feature_extractor(trainer, training_args)
        finalize_and_save_results(training_args, val_results, test_results, detailed_report)

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        try:
            if trainer is not None and training_args is not None:
                print("Saving partial model and results...")
                trainer.save_model(training_args.output_dir)
                partial_results = {
                    "validation": val_results,
                    "test": test_results,
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
