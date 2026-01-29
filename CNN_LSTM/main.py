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
from cnn_lstm_model import WeightedTrainer, compute_metrics, CNNLSTMModel, compute_class_weights
from transformers import TrainingArguments, SchedulerType, EarlyStoppingCallback
from common import TRAIN_SET, CALIBRATION_SET
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mainLoop(epoch: int):
    use_calibration_on_eval = True
    textFlag= ""
    if use_calibration_on_eval:
        textFlag = "TrainEvalOnCalibrationSet"

    # 1) Set seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

    dataset = load_dataset("audiofolder", data_dir="C:/Users/tapio/Desktop/Aineistot/TrainingDatasets")
    train_data = dataset["train"]
    validation_data = dataset["validation"]

    print(f"Loaded {len(train_data)} training examples")
    print(f"Loaded {len(validation_data)} validation examples")

    ds = {
        "train": preprocess_split(train_data, augment=True),
        "validation": preprocess_split(validation_data, augment=False),
    }
    del dataset, train_data, validation_data
    gc.collect()
    cuda_status = torch.cuda.is_available()

    prepared_train = ds["train"]
    desktop_path = Path.home() / "Desktop"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = desktop_path / f"cnn_lstm_model_{timestamp} + {epoch}epochs + {textFlag}"
    model_dir.mkdir(exist_ok=True)

    #class_weights = compute_class_weights(ds["train"])
    class_weights = torch.tensor([0.4, 3.3], device=device)
    print(f"Using manually adjusted class weights: {class_weights.tolist()}")

    final_training_args = TrainingArguments(
        output_dir=str(model_dir),
        logging_dir="./final_model/tensorboard",
        report_to="tensorboard",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=epoch,
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
        model=CNNLSTMModel(num_labels=2).to(device),
        args=final_training_args,
        train_dataset=prepared_train,
        data_collator=collate_fn,
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
        class_weights = class_weights,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=40)]
    )
    final_trainer.train()
    final_trainer.save_model(str(model_dir))

    final_validation_metrics = final_trainer.evaluate()
    print("Al-Emadi Eval metrics: ", final_validation_metrics)

    # Could also save the toroch model for others to eval. But we still have the safetensor file along with model details so it is ok.

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
        },
        "class_weights": class_weights.cpu().tolist(),
        "training_config": training_args_dict
    }

    with open(model_dir / "training_summary.json", "w") as f:
        json.dump(training_summary, f, indent=2, default=str)

def main():
    EpochList = [1000]
    for epoch in EpochList:
        print(f"\n\n=== RUN with {epoch} epochs ===")
        mainLoop(epoch)

if __name__ == "__main__":
    main()
