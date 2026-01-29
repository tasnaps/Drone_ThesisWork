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
    average_precision_score, precision_score, recall_score, f1_score
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

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = F.cross_entropy(logits, labels, weight=self.class_weights, label_smoothing=0.2)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Debug: Check class distribution
    print(f"Label distribution: {np.bincount(labels)}")
    print(f"Prediction distribution: {np.bincount(predictions)}")
    print(f"Unique labels: {np.unique(labels)}")
    print(f"Unique predictions: {np.unique(predictions)}")

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
def compute_class_weights(dataset):
    """Computes class weights based on the inverse frequency of labels in the entire dataset."""
    labels = [item['labels'] for item in dataset]
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long))
    weights = 1.0 / (counts.float() + 1e-8) # Add epsilon to avoid division by zero
    weights = weights / weights.sum() * len(weights) # Normalize
    print(f"Full dataset class counts: {counts}")
    print(f"Calculated class weights: {weights}")
    return weights.to(device)