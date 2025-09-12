import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import TrainingArguments, SchedulerType, EarlyStoppingCallback
from trainer import WeightedTrainer, compute_metrics, compute_class_weights
from data import collate_fn
from utils import log_misclassifications

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cross_validate_cnn_lstm(
        raw_train, prepared_train, model_init_fn,
        k=5, epochs_per_fold=5, batch_size=8, accum_steps=2, seed=42
    ):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    y = np.array(raw_train["label"])
    X = np.arange(len(y))
    fold_results = []
    for fold,(ti,vi) in enumerate(skf.split(X,y),1):
        args = TrainingArguments(
            output_dir=f"./cv_fold{fold}",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=accum_steps,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs_per_fold,
            fp16=True,
            logging_strategy="epoch",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-4,
            overwrite_output_dir=True,
            # === new scheduler & early-stop settings ===
            lr_scheduler_type=SchedulerType.COSINE,
            warmup_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        train_ds = prepared_train.select(ti)
        val_ds   = prepared_train.select(vi)
        weights  = compute_class_weights(raw_train.select(ti))
        trainer  = WeightedTrainer(
            class_weights = weights,
            model = model_init_fn().to(device),
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
        trainer.train()
        history = trainer.state.log_history
        log_misclassifications(trainer, val_ds, f"cv_fold_{fold}_misclassified.csv", best_thresh=0.5)
        final_ap= next(l["eval_avg_precision"] for l in reversed(history) if "eval_avg_precision" in l)
        fold_results.append({"history":history, "pr_auc":final_ap})
        del trainer
        torch.cuda.empty_cache()
    return fold_results