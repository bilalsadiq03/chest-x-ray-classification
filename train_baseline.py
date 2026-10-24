import sys
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score
)

sys.path.insert(0, str(Path.cwd()))
from src.data.dataset import ChestXrayDataset

SEED = 42
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-3
NUM_WORKERS = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(6)

OUT = Path("outputs")
CHECKPOINTS = Path("checkpoints")
OUT.mkdir(exist_ok=True)
CHECKPOINTS.mkdir(exist_ok=True)

train_ds = ChestXrayDataset("outputs/splits/train.csv", training=True)
val_ds = ChestXrayDataset("outputs/splits/validation.csv", training=False)
test_ds = ChestXrayDataset("outputs/splits/test.csv", training=False)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS
)

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
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(96, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def evaluate(model, loader):
    model.eval()
    total_loss, count = 0.0, 0
    labels, probs = [], []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)
            p = torch.softmax(logits, dim=1)[:, 1]
            total_loss += loss.item() * len(y)
            count += len(y)
            labels.extend(y.tolist())
            probs.extend(p.tolist())

    pred = (np.asarray(probs) >= 0.5).astype(int)
    acc = float(np.mean(pred == np.asarray(labels)))
    auc = float(roc_auc_score(labels, probs))
    return total_loss / count, acc, auc, labels, probs, pred

model = SmallCNN()

# Inverse-frequency class weighting, computed only from training labels.
counts = np.bincount(
    [int(r["label_id"]) for r in train_ds.records], minlength=2
)
weights = counts.sum() / (2.0 * counts)
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(weights, dtype=torch.float32)
)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
)

print("Training samples:", len(train_ds))
print("Validation samples:", len(val_ds))
print("Test samples:", len(test_ds))
print("Training class counts [NORMAL, PNEUMONIA]:", counts.tolist())
print("Class weights:", weights.tolist())

best_val_auc = -1.0
best_epoch = None
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, seen = 0.0, 0

    for x, y in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(y)
        seen += len(y)

    val_loss, val_acc, val_auc, *_ = evaluate(model, val_loader)
    row = {
        "epoch": epoch,
        "train_loss": running_loss / seen,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_roc_auc": val_auc
    }
    history.append(row)
    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"train_loss={row['train_loss']:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.4f} | val_auc={val_auc:.4f}"
    )

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_epoch = epoch
        torch.save(model.state_dict(), CHECKPOINTS / "best_cnn.pth")

with (OUT / "training_history.json").open("w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

# Load the checkpoint selected using validation ROC-AUC.
model.load_state_dict(torch.load(
    CHECKPOINTS / "best_cnn.pth", map_location="cpu", weights_only=True
))

test_loss, test_acc, test_auc, y_true, y_prob, y_pred = evaluate(
    model, test_loader
)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
report = classification_report(
    y_true, y_pred,
    labels=[0, 1],
    target_names=["NORMAL", "PNEUMONIA"],
    output_dict=True,
    zero_division=0
)

print("\nBest validation epoch:", best_epoch)
print("Test loss:", round(test_loss, 4))
print("Test accuracy:", round(test_acc, 4))
print("Test ROC-AUC:", round(test_auc, 4))
print("Test confusion matrix (rows=true, columns=predicted):")
print(cm)
print("\nClassification report:")
print(classification_report(
    y_true, y_pred,
    labels=[0, 1],
    target_names=["NORMAL", "PNEUMONIA"],
    zero_division=0
))

results = {
    "best_validation_epoch": best_epoch,
    "best_validation_roc_auc": best_val_auc,
    "test_loss": test_loss,
    "test_accuracy": test_acc,
    "test_roc_auc": test_auc,
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "test_labels": y_true,
    "test_pneumonia_probabilities": y_prob,
    "test_predictions": y_pred,
    "warning": "Custom CNN trained from scratch on CPU; not a pretrained medical model."
}

with (OUT / "baseline_results.json").open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=lambda obj: obj.tolist()
               if isinstance(obj, np.ndarray) else float(obj))

print("\nSaved checkpoint: checkpoints/best_cnn.pth")
print("Saved results: outputs/baseline_results.json")
