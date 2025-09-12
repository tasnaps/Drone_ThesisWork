"""Contains the model defition and helper functions."""
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from collections import Counter
from torch.utils.checkpoint import checkpoint_sequential
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import Trainer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    average_precision_score
)
# Import shared label mappings
from common import get_label_mappings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label2id, id2label = get_label_mappings()

class CNNLSTMModel(nn.Module):
    def __init__(self, num_labels: int, hidden_size: int = 128, lstm_layers: int = 1):
        super().__init__()
        # CNN to extract features from spectrogram
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # (freq/2, time/2)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))   # (freq/4, time/4)
        )
        # LSTM to model temporal sequence
        self.hidden_size = hidden_size
        # After CNN: channels=64, freq'=32 (128/4)
        self.lstm = nn.LSTM(
            input_size=64 * (128 // 4),
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, pixel_values, labels=None):
        # pixel_values: (batch, 1, n_mels, time_steps)
        modules = list(self.cnn.children())
        seg1 = nn.Sequential(*modules[:4])
        seg2 = nn.Sequential(*modules[4:])

        x = checkpoint_sequential([seg1, seg2], 2, pixel_values)
        b, c, f, t = x.size()
        # reshape for LSTM: (batch, time', features)
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        lstm_out, _ = self.lstm(x)            # (batch, time', hidden*2)
        final = lstm_out[:, -1, :]           # (batch, hidden*2)
        logits = self.classifier(final)      # (batch, num_labels)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)

def compute_class_weights(dataset) -> torch.Tensor:
    counts = Counter(dataset["label"])
    total = sum(counts.values())
    num_classes = len(counts)
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
    return torch.tensor(weights, device=device)

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device).to(logits.dtype)
        loss = F.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    # unpack
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    # handle HF tuple
    if isinstance(logits, tuple):
        logits = logits[0]

    preds = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:,1]

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    # PR-AUC
    ap = average_precision_score(labels, probs)

    return {
        "accuracy":    acc,
        "precision":   precision,
        "recall":      recall,
        "f1":          f1,
        "avg_precision": ap
    }
