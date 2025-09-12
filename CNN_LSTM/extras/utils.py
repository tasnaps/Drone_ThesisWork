import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os, csv
import torch
from data import id2label
import numpy as np

def log_misclassifications(trainer, dataset, filepath, best_thresh=0.5):
    preds = trainer.predict(dataset)
    probs = torch.softmax(torch.tensor(preds.predictions), dim=-1).cpu().numpy()[:,1]
    pred_labels = (probs > best_thresh).astype(int)

    with open(filepath, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "true_label", "pred_label", "prob"])
        for i, example in enumerate(dataset):
            if pred_labels[i] != preds.label_ids[i]:
                writer.writerow([
                    example["path"],
                    id2label[preds.label_ids[i]],
                    id2label[pred_labels[i]],
                    probs[i]
                ])

def plot_cv_histories(fold_results, metrics=("eval_loss","eval_f1","eval_accuracy","eval_avg_precision")):
    for m in metrics:
        plt.figure();
        for i,fr in enumerate(fold_results,1):
            h=fr["history"]
            xs=[l["epoch"] for l in h if m in l]
            ys=[l[m] for l in h if m in l]
            plt.plot(xs,ys,label=f"Fold {i}")
        plt.xlabel("Epoch");plt.ylabel(m);plt.legend();plt.tight_layout();plt.show()
    # bar chart of final AUCs
    aucs=[fr["pr_auc"] for fr in fold_results]
    plt.figure();
    plt.bar(range(1,len(aucs)+1),aucs, tick_label=[f"Fold {i}" for i in range(1,len(aucs)+1)])
    plt.ylabel("PR-AUC");plt.ylim(0,1);plt.tight_layout();plt.show()

def save_cv_results(fold_results, out_dir="cv_results"):
    os.makedirs(out_dir, exist_ok=True)
    # 1) per-fold, per-epoch
    per_epoch_file = os.path.join(out_dir, "per_epoch_metrics.csv")
    with open(per_epoch_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold", "epoch",
            "eval_loss","eval_accuracy","eval_precision",
            "eval_recall","eval_f1","eval_avg_precision"
        ])
        for fold_idx, fr in enumerate(fold_results, start=1):
            for log in fr["history"]:
                if "epoch" in log:
                    writer.writerow([
                        fold_idx,
                        log["epoch"],
                        log.get("eval_loss",""),
                        log.get("eval_accuracy",""),
                        log.get("eval_precision",""),
                        log.get("eval_recall",""),
                        log.get("eval_f1",""),
                        log.get("eval_avg_precision","")
                    ])
    print(f"Saved per-epoch CV metrics to {per_epoch_file}")

    # 2) final summary per fold
    summary_file = os.path.join(out_dir, "fold_summary.csv")
    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "final_pr_auc"])
        for fold_idx, fr in enumerate(fold_results, start=1):
            writer.writerow([fold_idx, fr["pr_auc"]])
    print(f"Saved per-fold PR-AUC summary to {summary_file}")

def plot_confusion_matrix(trainer, dataset, best_thresh=0.5):
    preds = trainer.predict(dataset)
    probs = torch.softmax(torch.tensor(preds.predictions), dim=-1).cpu().numpy()[:,1]
    pred_labels = (probs > best_thresh).astype(int)

    cm = confusion_matrix(preds.label_ids, pred_labels)
    report = classification_report(preds.label_ids, pred_labels, target_names=list(id2label.values()))

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=id2label.values(), yticklabels=id2label.values())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    with open("classification_report.txt", "w") as f:
        f.write(report)

def plot_threshold_diagnostics(trainer, ds, name, bayes_thresh):
    """
    Plot diagnostic histograms and Pmiss curves for a given dataset.

    - Histogram is color-coded: blue for negatives (label=0), red for positives (label=1).
    - Y-axis is logarithmic to emphasize low-frequency bins.
    """
    # 1) get model outputs
    out    = trainer.predict(ds)
    logits = out.predictions
    labels = out.label_ids           # array of 0/1
    probs  = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()[:, 1]

    # 2) histogram by true label with log scale
    plt.figure(figsize=(6, 4))
    plt.hist(
        probs[labels == 0],
        bins=40,
        alpha=0.6,
        color='blue',
        label='Other (true label = 0)'
    )
    plt.hist(
        probs[labels == 1],
        bins=40,
        alpha=0.6,
        color='red',
        label='Drone (true label = 1)'
    )
    plt.axvline(
        bayes_thresh,
        linestyle="--",
        linewidth=2,
        color="black",
        label=f"Bayes thresh = {bayes_thresh:.2f}"
    )
    plt.yscale('log')
    plt.xlabel("Predicted probability")
    plt.ylabel("Count (log scale)")
    plt.title(f"{name}: P(drone|x) histogram")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) Pmiss vs. threshold
    ths    = np.linspace(0, 1, 101)
    p_miss = [((probs < t) & (labels == 1)).mean() for t in ths]

    plt.figure(figsize=(6, 4))
    plt.plot(
        ths,
        p_miss,
        linewidth=2,
        color='red',
        label='Pmiss(t)'
    )
    plt.axvline(
        bayes_thresh,
        linestyle="--",
        color="black",
        label=f"Bayes thresh = {bayes_thresh:.2f}"
    )
    plt.title(f"{name}: Pmiss(t) vs threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Pmiss(t)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # 4) scatter by sample index
    plt.figure(figsize=(8, 4))
    indices = np.arange(len(probs))
    # negatives
    plt.scatter(
        indices[labels == 0],
        probs[labels == 0],
        alpha=0.4,
        label='Other (true=0)',
        color='blue',
        s=10
    )
    # positives
    plt.scatter(
        indices[labels == 1],
        probs[labels == 1],
        alpha=0.4,
        label='Drone (true=1)',
        color='red',
        s=10
    )
    plt.axhline(
        bayes_thresh,
        linestyle='--',
        color='black',
        linewidth=2,
        label=f'Threshold = {bayes_thresh:.2f}'
    )
    plt.xlabel('Sample index')
    plt.ylabel('P(drone)')
    plt.title(f'{name}: Predictions vs. Threshold')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()