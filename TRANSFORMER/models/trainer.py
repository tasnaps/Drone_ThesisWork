import torch
import torch.nn.functional as F
from transformers import Trainer
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
    fbeta_score,
)
from collections import Counter

def compute_class_weights(dataset) -> torch.Tensor:
    """
    Compute class weights = total/(n_classes * count_i)
    for a HuggingFace DatasetDict split that has a 'label' column.
    """
    counts = Counter(dataset["label"])
    total = sum(counts.values())
    num_classes = len(counts)
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
    return torch.tensor(weights)

# weighted cross‐entropy
def weighted_loss_fn(logits, labels, weights):
    w = weights.to(logits.device).to(logits.dtype)
    return F.cross_entropy(logits, labels, weight=w)

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = weighted_loss_fn(outputs.logits, labels, self.class_weights)
        return (loss, outputs) if return_outputs else loss

# metrics including PR‐AUC
def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = logits.argmax(-1)
    probs = softmax(logits, axis=-1)[:,1]
    acc = accuracy_score(labels, preds)
    roc_auc = roc_auc_score(labels, probs)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary")
    ap = average_precision_score(labels, probs)
    # Add F2 score so it is available as eval_f2 for best model selection
    f2 = fbeta_score(labels, preds, beta=2.0, average="binary")
    return {"accuracy":acc, "precision":p, "recall":r, "f1":f, "f2": f2, "avg_precision":ap, "roc_auc":roc_auc}