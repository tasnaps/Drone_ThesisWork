import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from collections import Counter
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_dataset, Audio, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)
from transformers import TrainingArguments, Trainer
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_label_mappings():
    label2id = {"unknown": 0, "yes_drone": 1}
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label

label2id, id2label = get_label_mappings()

def load_and_split(
    data_dir: str,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42
) -> DatasetDict:
    raw = load_dataset("audiofolder", data_dir=data_dir)
    tmp = raw["train"].train_test_split(
        test_size=val_size + test_size,
        seed=seed,
        stratify_by_column="label"
    )
    vt = tmp["test"].train_test_split(
        test_size=test_size / (val_size + test_size),
        seed=seed,
        stratify_by_column="label"
    )
    return DatasetDict({
        "train": tmp["train"],
        "validation": vt["train"],
        "test": vt["test"]
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

def compute_class_weights(dataset) -> torch.Tensor:
    counts = Counter(dataset["label"])
    total = sum(counts.values())
    num_classes = len(counts)
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)

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

def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

    # Set output directory to desktop
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    output_dir = os.path.join(desktop_path, "resnet34-audio")

    data_dir = "C:/Gradu Juttui/Datasets/DroneAudioDataset_Saraalemadi/Binary_Drone_Audio"
    ds = load_and_split(data_dir)

    # log class distributions
    for split in ds:
        cnt = Counter(ds[split]["label"])
        print(f"{split.upper()}:", {id2label[k]: v for k, v in cnt.items()})

    class_weights = compute_class_weights(ds["train"])

    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    for split in ["train", "validation", "test"]:
        ds[split] = ds[split].map(
            prepare_batch,
            remove_columns=["audio", "label"],
            num_proc=os.cpu_count(),
            desc=f"Prep {split}"
        )

    model = ResNetForAudioClassification(num_labels=len(label2id), freeze_blocks=False)
    model.to(device)

    # calculate total training steps for warmup
    per_device_bs = 16
    num_epochs = 100
    total_steps = (len(ds["train"]) // per_device_bs) * num_epochs

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
        learning_rate=3e-4,
        warmup_steps=int(0.1 * total_steps),
        logging_steps=100,
        save_total_limit=2,
        save_strategy="epoch",
        dataloader_num_workers=min(os.cpu_count()-1, 4),
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

    trainer.train()
    print("Validation:", trainer.evaluate())
    print("Test:", trainer.evaluate(ds["test"]))

    # threshold calibration
    val_out = trainer.predict(ds["validation"])
    val_probs = torch.softmax(torch.tensor(val_out.predictions), dim=-1).cpu().numpy()[:,1]
    precs, recs, ths = precision_recall_curve(val_out.label_ids, val_probs)
    f1s = 2 * precs * recs / (precs + recs + 1e-8)
    best_t = ths[np.nanargmax(f1s)]
    print(f"Best threshold: {best_t:.3f}")

    test_out = trainer.predict(ds["test"])
    test_probs = torch.softmax(torch.tensor(test_out.predictions), dim=-1).cpu().numpy()[:,1]
    test_preds = (test_probs > best_t).astype(int)
    print("Thresholded Test Metrics:",
        accuracy_score(test_out.label_ids, test_preds),
        precision_recall_fscore_support(test_out.label_ids, test_preds, average="binary", zero_division=0)
    )
    print("Confusion:\n", confusion_matrix(test_out.label_ids, test_preds))
    print("Report:\n", classification_report(test_out.label_ids, test_preds,
          target_names=[id2label[i] for i in sorted(id2label)]))

    trainer.save_model(output_dir)
    print(f"Model saved to: {output_dir}")

if __name__ == "__main__":
    main()
