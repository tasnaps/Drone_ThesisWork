"""Training script for the CNN-LSTM model- remember to edit the output dir if you dont want it to go to Desktop.
 The datasets are defined in the common.py"""
import datetime
import random
import numpy as np
import gc
import warnings
import torch
import json
from datasets import load_dataset
from pathlib import Path
from data import load_and_split, preprocess_split, collate_fn
from cnn_lstm_model import WeightedTrainer, compute_metrics, compute_class_weights, CNNLSTMModel
from transformers import TrainingArguments, SchedulerType
from common import TRAIN_SET

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1) Set seed
    seed=42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

    raw_ds = load_and_split(TRAIN_SET, val_size=0.1, test_size=0.1, seed=seed)
    class_weights = compute_class_weights(raw_ds["train"])

    # Preprocess splits
    ds = {
        "train": preprocess_split(raw_ds["train"], augment=True),
        "validation": preprocess_split(raw_ds["validation"], augment=False),
        "test": preprocess_split(raw_ds["test"], augment=False)
    }
    del raw_ds
    gc.collect()
    prepared_train = ds["train"]

    cuda_status = torch.cuda.is_available()

    desktop_path = Path.home() / "Desktop"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = desktop_path / f"cnn_lstm_model_{timestamp}"
    model_dir.mkdir(exist_ok=True)

    final_training_args = TrainingArguments(
        output_dir=str(model_dir),
        logging_dir="./final_model/tensorboard",
        report_to="tensorboard",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        fp16=cuda_status,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=3e-4,
        lr_scheduler_type=SchedulerType.COSINE_WITH_RESTARTS,
        warmup_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
    )

    # paste this to callbacks if you want early stopping[EarlyStoppingCallback(early_stopping_patience=3)]
    final_trainer = WeightedTrainer(
        class_weights=class_weights,
        model=CNNLSTMModel(num_labels=2).to(device),
        args=final_training_args,
        train_dataset=prepared_train,
        data_collator=collate_fn,
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
        callbacks=None
    )
    final_trainer.train()
    final_trainer.save_model(str(model_dir))
    #Could also save the toroch model for others to eval. But we still have the safetensor file along with model details so it is ok.

    # Save training arguments as readable JSON
    training_args_dict = final_training_args.to_dict()
    with open(model_dir / "training_args.json", "w") as f:
        json.dump(training_args_dict, f, indent=2, default=str)

    # Also save a summary of the training run
    training_summary = {
        "model_type": "CNN-LSTM",
        "dataset": TRAIN_SET,
        "timestamp": timestamp,
        "seed": seed,
        "num_labels": 2,
        "cuda_available": cuda_status,
        "dataset_splits": {
            "train_size": len(ds["train"]),
            "validation_size": len(ds["validation"]),
            "test_size": len(ds["test"])
        },
        "class_weights": class_weights.cpu().tolist(),
        "training_config": training_args_dict
    }

    with open(model_dir / "training_summary.json", "w") as f:
        json.dump(training_summary, f, indent=2, default=str)

def load_audiofolder(path):
    """
    Load an audiofolder dataset and ensure a 'label' column exists.
    If none is found, assign label=1 (drone) to every example.
    """
    ds = load_dataset("audiofolder", data_dir=path)["train"]
    # if this folder has only one class, HF will omit 'label'
    if "label" not in ds.column_names:
        # add a column of ones (all positive examples)
        ds = ds.add_column("label", [1] * ds.num_rows)
    return ds

if __name__ == "__main__":
    main()
