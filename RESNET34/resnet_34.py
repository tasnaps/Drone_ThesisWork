import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from collections import Counter
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # silence TF INFO/WARNING
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_dataset, Audio, DatasetDict, Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)
from config import RUNS, TRAIN_SET
from transformers import TrainingArguments, Trainer, TrainerCallback
from torchvision import models

class EvalPrinterCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Print only numeric metrics to keep it clean
        clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        print("\nEval metrics:", clean)
        return control

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_label_mappings():
    label2id = {"unknown": 0, "yes_drone": 1}
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label

label2id, id2label = get_label_mappings()

def load_and_split(
    data_dir: str,
    val_size: float = 0.2,
    seed: int = 42
) -> DatasetDict:
    """
    Load an audio dataset from folders and split into train/validation with stratification fallback.
    """
    raw = load_dataset("audiofolder", data_dir=data_dir)
    if "label" in raw["train"].column_names:
        # Try stratified split first
        try:
            tmp = raw["train"].train_test_split(
                test_size=val_size,
                seed=seed,
                stratify_by_column="label"
            )
        except (ValueError, TypeError):
            # Fall back to regular split if stratification fails
            tmp = raw["train"].train_test_split(
                test_size=val_size,
                seed=seed
            )
    else:
        # No label column, use regular split
        tmp = raw["train"].train_test_split(
            test_size=val_size,
            seed=seed
        )

    return DatasetDict({
        "train": tmp["train"],
        "validation": tmp["test"],
    })

class ResNetForAudioClassification(nn.Module):
    def __init__(self, num_labels: int, freeze_blocks: bool = False):
        super().__init__()
        # load pretrained ResNet-34
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # replace final classifier
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_labels)
        self.num_labels = num_labels

        # we can choose to only train the last block
        if freeze_blocks:
            for name, module in self.resnet.named_modules():
                if not name.startswith("layer4") and "fc" not in name:
                    # Freeze parameters
                    for param in module.parameters():
                        param.requires_grad = False

                    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        module.eval()

    def forward(self, pixel_values, labels=None):
        logits = self.resnet(pixel_values)  # (batch, num_labels)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

spec_extractor = T.MelSpectrogram(
    sample_rate=16000,
    n_mels=128,
    n_fft=1024,
    hop_length=256,
)
# ImageNet normalization for ResNet-34
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

def normalize_imagenet(tensor):
    """Apply ImageNet normalization to 3-channel tensor"""
    # tensor shape: (3, H, W)
    mean = IMAGENET_MEAN.view(3, 1, 1)
    std = IMAGENET_STD.view(3, 1, 1)
    return (tensor - mean) / std

def preprocess_audio_to_resnet_input(waveform_array, sample_rate=16000, target_duration=1.0):
    """
    Shared preprocessing function for both training and evaluation.
    Converts audio to ResNet-ready input with consistent 1-second scale.
    """
    waveform = torch.tensor(waveform_array, dtype=torch.float32)

    # Ensure mono
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)

    # Pad/trim to exactly 1.0 second (16,000 samples)
    target_samples = int(target_duration * sample_rate)
    if waveform.shape[-1] > target_samples:
        waveform = waveform[:target_samples]
    elif waveform.shape[-1] < target_samples:
        pad_length = target_samples - waveform.shape[-1]
        waveform = F.pad(waveform, (0, pad_length), mode='constant', value=0)

    # Apply minimum padding for mel spectrogram if needed
    if waveform.shape[-1] < 1024:
        pad = torch.zeros(1024 - waveform.shape[-1], dtype=torch.float32)
        waveform = torch.cat([waveform, pad], dim=-1)

    # Extract mel spectrogram
    mel = spec_extractor(waveform)  # (n_mels=128, time_steps)
    mel = torch.log1p(mel)

    # Convert to 3-channel image and resize to 224×224
    img = mel.unsqueeze(0).repeat(3, 1, 1)  # (3, 128, time_steps)
    img = F.interpolate(
        img.unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)  # (3, 224, 224)

    # Apply ImageNet normalization
    img = normalize_imagenet(img)

    return img

def prepare_batch(batch):
    """Updated to use shared preprocessor and 1-second clips"""
    img = preprocess_audio_to_resnet_input(batch["audio"]["array"])
    batch["pixel_values"] = img.numpy()
    batch["labels"] = batch["label"]
    return batch

def collate_fn(features):
    pixel_vals = torch.stack([
        torch.tensor(f["pixel_values"], dtype=torch.float32)
        for f in features
    ])
    labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
    return {"pixel_values": pixel_vals, "labels": labels}

#def compute_class_weights(dataset) -> torch.Tensor:
#    counts = Counter(dataset["label"])
#    total = sum(counts.values())
#    num_classes = len(counts)
#    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
#    return torch.tensor(weights, dtype=torch.float32)

def weighted_loss_fn(logits, labels, weights):
    w = weights.to(logits.device).to(logits.dtype)
    return F.cross_entropy(logits, labels, weight=w)

class WeightedTrainer(Trainer):
    def __init__(self, *, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = weighted_loss_fn(outputs.logits, labels, self.class_weights)

        # Scale loss by the number of items in batch when gradient accumulation is used
        if num_items_in_batch is not None:
            loss = loss * num_items_in_batch / labels.shape[0]

        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1) if isinstance(logits, torch.Tensor) else np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def compute_class_weights(ds: Dataset) -> torch.Tensor:
    """
    Computes class weights from a HF Dataset. Works with either raw ('label') or
    preprocessed ('labels') datasets. Normalized to sum to num_classes.
    """
    col = "labels" if "labels" in ds.column_names else "label"
    labels = ds[col]  # list of ints
    num_classes = len(label2id)
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes)
    weights = 1.0 / (counts.float() + 1e-8)
    weights = weights / weights.sum() * num_classes
    print(f"Class counts: {counts.tolist()}")
    print(f"Class weights: {weights.tolist()}")
    return weights.to(device)

def og_data_splitting(per_device_bs=16, num_epochs=20):
    data_dir = TRAIN_SET
    ds = load_and_split(data_dir, val_size=0.2)

    # log class distributions
    for split in ds:
        cnt = Counter(ds[split]["label"])
        print(f"{split.upper()}:", {id2label[k]: v for k, v in cnt.items()})

    class_weights = compute_class_weights(ds["train"])

    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    for split in ["train", "validation"]:
        ds[split] = ds[split].map(
            prepare_batch,
            remove_columns=["audio", "label"],
            num_proc=os.cpu_count(),
            desc=f"Prep {split}"
        )
    total_steps = (len(ds["train"]) // per_device_bs) * num_epochs
    total_steps = int(0.1 * total_steps)

    return ds, class_weights, total_steps

def prepare_batch_batched(batch):
    """
    Batched preprocessor for Hugging Face Datasets with Audio feature.
    Handles both list-of-dicts and dict-of-lists forms produced by batched mapping.
    """
    aud = batch["audio"]

    # Get list of raw waveforms
    if isinstance(aud, dict) and "array" in aud:
        arrays = aud["array"]  # dict of lists: {"array": [...], "sampling_rate": [...]}
    else:
        arrays = [a["array"] for a in aud]  # list of dicts: [{"array": ..., "sampling_rate": ...}, ...]

    pixel_values = [preprocess_audio_to_resnet_input(arr).numpy() for arr in arrays]
    out = {"pixel_values": pixel_values}

    if "label" in batch:
        # HF Datasets will keep this as a list aligned with the batch
        out["labels"] = batch["label"]

    return out

def preprocess_split(
    ds: Dataset,
    augment: bool = False
) -> Dataset:
    """
    Apply feature extraction and optional augmentation to an entire dataset in batched mode.
    """
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    remove_cols = ["audio"]
    if "label" in ds.column_names:
        remove_cols.append("label")

    return ds.map(
        prepare_batch_batched,           # <- use the batched preprocessor
        remove_columns=remove_cols,
        batched=True,
        batch_size=128,                  # keep memory reasonable
        num_proc=min(os.cpu_count() or 1, 6),
        desc="Preprocessing (batched)"
    )

def alt_data_split_for_custom_validation():
    dataset = load_dataset("audiofolder", data_dir="C:/Users/tapio/Desktop/Aineistot/TrainingDatasets")
    train_data = dataset["train"]
    validation_data = dataset["validation"]
    ds = {
        "train": preprocess_split(train_data, augment=False),
        "validation": preprocess_split(validation_data, augment=False),
    }
    return ds

def main_loop(epoch: int):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

    # Set output directory to desktop
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    output_dir = os.path.join(desktop_path, "resnet34-audio-model-calibrationsetEVAL", f"run-{epoch}")




    model = ResNetForAudioClassification(num_labels=len(label2id), freeze_blocks=False)
    model.to(device)
    ds = alt_data_split_for_custom_validation()
    per_device_bs = 16
    num_epochs = epoch
    total_steps = (len(ds["train"]) // per_device_bs) * num_epochs
    total_steps = int(0.1 * total_steps)
    class_weights = compute_class_weights(ds["train"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,
        logging_strategy="epoch",
        disable_tqdm=True,
        learning_rate=3e-4,
        warmup_steps=total_steps,
        logging_steps=100,
        save_total_limit=2,
        save_strategy="epoch",
        report_to="none",
        log_level="info",
        dataloader_num_workers=min(os.cpu_count() - 1, 4),
        dataloader_pin_memory=True,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
    )
    trainer.add_callback(EvalPrinterCallback())
    trainer.train()

    # Validation metrics:
    final_val_metrics = trainer.evaluate()
    print("Validation Metrics:", final_val_metrics)

    trainer.save_model(output_dir)
    print(f"Model saved to: {output_dir}")

def main():
    for run in [1000]:
        print(f"\n\n=== RUN {run} ===")
        main_loop(run)

if __name__ == "__main__":
    main()
