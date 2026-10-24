import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

sys.path.insert(0, str(Path.cwd()))
from src.data.dataset import ChestXrayDataset

torch.set_num_threads(6)

CHECKPOINT = Path("checkpoints/best_cnn.pth")
MANIFEST = Path("outputs/splits/test.csv")
RESULTS = Path("outputs/baseline_results.json")
PREDICTIONS = Path("outputs/test_predictions.csv")

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
if not MANIFEST.exists():
    raise FileNotFoundError(f"Test manifest not found: {MANIFEST}")

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(96, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

dataset = ChestXrayDataset(str(MANIFEST), training=False)
loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

model = SmallCNN()
model.load_state_dict(
    torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
)
model.eval()

all_labels = []
all_probabilities = []
all_predictions = []
total_loss = 0.0
count = 0
criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for images, labels in loader:
        logits = model(images)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = (probabilities >= 0.5).long()

        total_loss += loss.item() * len(labels)
        count += len(labels)
        all_labels.extend(labels.tolist())
        all_probabilities.extend(probabilities.tolist())
        all_predictions.extend(predictions.tolist())

y_true = np.asarray(all_labels, dtype=int)
y_prob = np.asarray(all_probabilities, dtype=float)
y_pred = np.asarray(all_predictions, dtype=int)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["NORMAL", "PNEUMONIA"],
    output_dict=True,
    zero_division=0,
)

history_path = Path("outputs/training_history.json")
best_epoch = None
best_val_auc = None

if history_path.exists():
    with history_path.open(encoding="utf-8") as f:
        history = json.load(f)
    if history:
        best_row = max(history, key=lambda row: row["val_roc_auc"])
        best_epoch = best_row["epoch"]
        best_val_auc = best_row["val_roc_auc"]

results = {
    "best_validation_epoch": best_epoch,
    "best_validation_roc_auc": best_val_auc,
    "test_loss": float(total_loss / count),
    "test_accuracy": float(np.mean(y_true == y_pred)),
    "test_roc_auc": float(roc_auc_score(y_true, y_prob)),
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "test_labels": y_true.tolist(),
    "test_pneumonia_probabilities": y_prob.tolist(),
    "test_predictions": y_pred.tolist(),
    "warning": (
        "Custom CNN trained from scratch on CPU; not a pretrained medical "
        "model. Metrics are from the held-out test split."
    ),
}

# Write to a temporary file first, then replace the damaged JSON.
temp_results = RESULTS.with_suffix(".tmp")
with temp_results.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, allow_nan=False)
temp_results.replace(RESULTS)

# Export predictions in the same order as the test manifest.
import csv

with PREDICTIONS.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "path", "true_label", "predicted_label",
            "pneumonia_probability", "correct", "case_type",
        ],
    )
    writer.writeheader()

    for record, truth, pred, prob in zip(
        dataset.records, y_true, y_pred, y_prob
    ):
        case_type = (
            "FALSE_NEGATIVE" if truth == 1 and pred == 0 else
            "FALSE_POSITIVE" if truth == 0 and pred == 1 else
            "TRUE_POSITIVE" if truth == 1 and pred == 1 else
            "TRUE_NEGATIVE"
        )
        writer.writerow({
            "path": record["path"],
            "true_label": "PNEUMONIA" if truth == 1 else "NORMAL",
            "predicted_label": "PNEUMONIA" if pred == 1 else "NORMAL",
            "pneumonia_probability": f"{prob:.8f}",
            "correct": int(truth == pred),
            "case_type": case_type,
        })

# Verify that the saved JSON is valid.
with RESULTS.open(encoding="utf-8") as f:
    json.load(f)

print("Recovery complete.")
print("Test samples:", len(y_true))
print("Test loss:", round(results["test_loss"], 4))
print("Test accuracy:", round(results["test_accuracy"], 4))
print("Test ROC-AUC:", round(results["test_roc_auc"], 4))
print("Confusion matrix:")
print(cm)
print("False negatives:", int(np.sum((y_true == 1) & (y_pred == 0))))
print("False positives:", int(np.sum((y_true == 0) & (y_pred == 1))))
print("Valid results saved:", RESULTS)
print("Predictions saved:", PREDICTIONS)
