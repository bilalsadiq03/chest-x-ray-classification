"""
scripts/recover_baseline.py — Re-run CNN inference on the held-out test set.

Usage (from project root):
    .venv\Scripts\python.exe scripts/recover_baseline.py

Loads checkpoints/best_cnn.pth, evaluates on outputs/splits/test.csv,
writes outputs/baseline_results.json and outputs/test_predictions.csv.
Safe to re-run: output files are written atomically.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import ChestXrayDataset
from src.models.cnn import build_model

torch.set_num_threads(6)

CHECKPOINT  = Path("checkpoints/best_cnn.pth")
MANIFEST    = Path("outputs/splits/test.csv")
RESULTS     = Path("outputs/baseline_results.json")
PREDICTIONS = Path("outputs/test_predictions.csv")

if not CHECKPOINT.exists():
    print(f"ERROR: Checkpoint not found: {CHECKPOINT}", file=sys.stderr)
    sys.exit(1)
if not MANIFEST.exists():
    print(f"ERROR: Test manifest not found: {MANIFEST}", file=sys.stderr)
    sys.exit(1)

dataset = ChestXrayDataset(str(MANIFEST), training=False)
loader  = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
model   = build_model(CHECKPOINT)

print(f"Checkpoint:    {CHECKPOINT}")
print(f"Test images:   {len(dataset)}")

all_labels:  list[int]   = []
all_probs:   list[float] = []
all_preds:   list[int]   = []
total_loss = 0.0
count      = 0
criterion  = nn.CrossEntropyLoss()

with torch.no_grad():
    for images, labels in loader:
        logits = model(images)
        loss   = criterion(logits, labels)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = (probs >= 0.5).long()

        total_loss += loss.item() * len(labels)
        count      += len(labels)
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())

y_true = np.asarray(all_labels, dtype=int)
y_prob = np.asarray(all_probs,  dtype=float)
y_pred = np.asarray(all_preds,  dtype=int)

cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
report = classification_report(
    y_true, y_pred,
    labels=[0, 1],
    target_names=["NORMAL", "PNEUMONIA"],
    output_dict=True,
    zero_division=0,
)

# Load training history if available
history_path = Path("outputs/training_history.json")
best_epoch, best_val_auc = None, None
if history_path.exists():
    with history_path.open(encoding="utf-8") as f:
        history = json.load(f)
    if history:
        best_row      = max(history, key=lambda r: r["val_roc_auc"])
        best_epoch    = best_row["epoch"]
        best_val_auc  = best_row["val_roc_auc"]

results = {
    "best_validation_epoch":    best_epoch,
    "best_validation_roc_auc":  best_val_auc,
    "test_loss":                float(total_loss / count),
    "test_accuracy":            float(np.mean(y_true == y_pred)),
    "test_roc_auc":             float(roc_auc_score(y_true, y_prob)),
    "confusion_matrix":         cm.tolist(),
    "classification_report":    report,
    "test_labels":              y_true.tolist(),
    "test_pneumonia_probabilities": y_prob.tolist(),
    "test_predictions":         y_pred.tolist(),
    "warning": (
        "Custom CNN trained from scratch on CPU; not a pretrained medical model. "
        "Metrics are from the held-out test split."
    ),
}

# Atomic write
tmp = RESULTS.with_suffix(".tmp")
with tmp.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, allow_nan=False)
tmp.replace(RESULTS)

# Verify
with RESULTS.open(encoding="utf-8") as f:
    json.load(f)

# Export per-image predictions
with PREDICTIONS.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["path", "true_label", "predicted_label",
                    "pneumonia_probability", "correct", "case_type"],
    )
    writer.writeheader()
    for record, truth, pred, prob in zip(dataset.records, y_true, y_pred, y_prob):
        case_type = (
            "FALSE_NEGATIVE" if truth == 1 and pred == 0 else
            "FALSE_POSITIVE" if truth == 0 and pred == 1 else
            "TRUE_POSITIVE"  if truth == 1 and pred == 1 else
            "TRUE_NEGATIVE"
        )
        writer.writerow({
            "path":                 record["path"],
            "true_label":          "PNEUMONIA" if truth == 1 else "NORMAL",
            "predicted_label":     "PNEUMONIA" if pred  == 1 else "NORMAL",
            "pneumonia_probability": f"{prob:.8f}",
            "correct":             int(truth == pred),
            "case_type":           case_type,
        })

print("\nRecovery complete.")
print(f"Test samples:      {len(y_true)}")
print(f"Test loss:         {results['test_loss']:.4f}")
print(f"Test accuracy:     {results['test_accuracy']:.4f}")
print(f"Test ROC-AUC:      {results['test_roc_auc']:.4f}")
print("Confusion matrix (rows=true, cols=predicted):")
print(cm)
print(f"False negatives:   {int(np.sum((y_true == 1) & (y_pred == 0)))}")
print(f"False positives:   {int(np.sum((y_true == 0) & (y_pred == 1)))}")
print(f"\nResults:           {RESULTS}")
print(f"Predictions:       {PREDICTIONS}")
