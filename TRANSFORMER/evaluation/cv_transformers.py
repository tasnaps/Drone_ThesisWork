import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import TrainingArguments, SchedulerType, EarlyStoppingCallback
from TRANSFORMER.models.trainer import WeightedTrainer, compute_metrics, compute_class_weights

def cross_validate_transformer(
    raw_train, prepared_train, model_init_fn, data_collator,
    k=5, epochs_per_fold=5, batch_size=8, accum_steps=2, seed=42
):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    y = np.array(raw_train["label"])
    X = np.arange(len(y))
    fold_results = []

    for fold, (ti, vi) in enumerate(skf.split(X, y), 1):
        args = TrainingArguments(
            output_dir=f"./cv_transformer_fold{fold}",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=accum_steps,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs_per_fold,
            fp16=torch.cuda.is_available(),
            logging_strategy="epoch",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-5,
            lr_scheduler_type=SchedulerType.COSINE_WITH_RESTARTS,
            warmup_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            overwrite_output_dir=True,
        )

        train_ds = prepared_train.select(ti)
        val_ds   = prepared_train.select(vi)
        weights  = compute_class_weights(raw_train.select(ti))

        trainer = WeightedTrainer(
            class_weights=weights,
            model=model_init_fn(),
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.train()
        history = trainer.state.log_history
        final_ap = next(
            log["eval_avg_precision"]
            for log in reversed(history)
            if "eval_avg_precision" in log
        )
        fold_results.append({"history": history, "pr_auc": final_ap})

        del trainer
        torch.cuda.empty_cache()

    return fold_results
