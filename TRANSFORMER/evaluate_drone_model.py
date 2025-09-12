#!/usr/bin/env python
import os
import argparse
import logging
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch
import evaluate
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc, roc_curve,
)
import matplotlib.pyplot as plt

# ─── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Argparse ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Load a fine-tuned Wav2Vec2 audio-classification model and evaluate on test set"
    )
    p.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory where the fine-tuned model & feature_extractor are saved",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to your Binary_Drone_Audio folder (with subdirs 'unknown'/'yes_drone')",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    p.add_argument(
        "--cuda",
        action="store_true",
        help="Use GPU if available",
    )
    return p.parse_args()

# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load dataset splits (re-do same train/val/test split logic)
    raw = load_dataset("audiofolder", data_dir=args.data_dir, split=None)
    splits = raw["train"].train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    tv = splits["test"].train_test_split(
        test_size=0.5, seed=42, stratify_by_column="label"
    )  # yields 10% val, 10% test
    ds: DatasetDict = DatasetDict({
        "test": tv["test"]
    })

    logger.info(f"Test split size: {len(ds['test'])}")
    counts = Counter(ds["test"]["label"])
    logger.info(f"Test distribution: {{'unknown': {counts[0]}, 'yes_drone': {counts[1]}}}")

    # 2. Load feature extractor & model
    feat = AutoFeatureExtractor.from_pretrained(args.model_dir)
    model = AutoModelForAudioClassification.from_pretrained(args.model_dir).to(device)

    # 3. Preprocess test set
    ds["test"] = ds["test"].cast_column("audio", Audio(sampling_rate=feat.sampling_rate))

    def preprocess(batch):
        audio = batch["audio"]
        inp = feat(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
        batch["input_values"] = inp.input_values[0]
        batch["labels"] = batch["label"]
        return batch

    ds["test"] = ds["test"].map(
        preprocess,
        remove_columns=["audio", "label"],
        num_proc=os.cpu_count(),
    )

    # 4. Run inference
    all_logits = []
    all_labels = []
    for i in range(0, len(ds["test"]), args.batch_size):
        batch = ds["test"][i: i + args.batch_size]  # dict of lists

        # 1) build a tensor batch of inputs
        input_list = [torch.tensor(arr) for arr in batch["input_values"]]
        inputs = pad_sequence(input_list, batch_first=True, padding_value=0.0).to(device)
        #inputs = torch.stack(input_list).to(device)


        # 2) labels into tensor (ground truth integers)
        label_list = batch["labels"]
        labels = torch.tensor(label_list).to(device)

        # 3) forward pass
        with torch.no_grad():
            logits = model(inputs).logits.cpu().numpy()

        all_logits.append(logits)
        all_labels.extend(label_list)
    all_logits = np.vstack(all_logits)
    all_labels = np.array(all_labels)

    # 5. Compute metrics
    preds = np.argmax(all_logits, axis=1)
    probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()[:, 1]

    # a) HF evaluate metrics
    acc_met = evaluate.load("accuracy")
    f1_met  = evaluate.load("f1")
    prec_met= evaluate.load("precision")
    rec_met = evaluate.load("recall")
    logger.info("Accuracy:  %.4f", acc_met.compute(predictions=preds, references=all_labels)["accuracy"])
    logger.info("Precision: %.4f", prec_met.compute(predictions=preds, references=all_labels)["precision"])
    logger.info("Recall:    %.4f", rec_met.compute(predictions=preds, references=all_labels)["recall"])
    logger.info("F1:        %.4f", f1_met.compute(predictions=preds, references=all_labels)["f1"])

    # b) Confusion matrix & report
    cm = confusion_matrix(all_labels, preds)
    logger.info("Confusion Matrix:\n%s", cm)
    logger.info("Classification Report:\n%s",
                classification_report(all_labels, preds, target_names=["unknown", "yes_drone"]))

    # c) ROC AUC & PR AUC
    roc_auc = roc_auc_score(all_labels, probs)
    prec, rec, thresh = precision_recall_curve(all_labels, probs)
    pr_auc = auc(rec, prec)
    logger.info("ROC AUC: %.4f", roc_auc)
    logger.info("PR  AUC: %.4f", pr_auc)

    # 6. Plot curves
    plt.figure(figsize=(6,4))
    fpr, tpr, _ = roc_curve(all_labels, probs)
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig("roc_curve.png")

    plt.figure(figsize=(6,4))
    plt.plot(rec, prec, label=f"PR AUC={pr_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curve"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig("pr_curve.png")

    logger.info("Saved ROC and PR curves as `roc_curve.png` and `pr_curve.png`")

if __name__ == "__main__":
    main()
