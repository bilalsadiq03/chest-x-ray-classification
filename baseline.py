from pathlib import Path
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

ROOT = Path("data/raw/chest_xray")
IMG_SIZE = 96
BATCH_SIZE = 16
EPOCHS = 3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

class ChestXrayDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            im = im.convert("L").resize((IMG_SIZE, IMG_SIZE))
            arr = np.asarray(im, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)
        return x, torch.tensor(label, dtype=torch.long)

def collect(split):
    samples = []
    for cls, label in [("NORMAL", 0), ("PNEUMONIA", 1)]:
        samples.extend(
            (p, label) for p in (ROOT / split / cls).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
    return samples

train_all = collect("train")
train_samples, val_samples = train_test_split(
    train_all,
    test_size=0.2,
    random_state=SEED,
    stratify=[y for _, y in train_all]
)
test_samples = collect("test")

train_loader = DataLoader(ChestXrayDataset(train_samples), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ChestXrayDataset(val_samples), batch_size=BATCH_SIZE)
test_loader = DataLoader(ChestXrayDataset(test_samples), batch_size=BATCH_SIZE)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(32, 2)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

model = SmallCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(y)

    model.eval()
    correct, count = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            count += len(y)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"loss={running_loss/len(train_samples):.4f} | "
          f"val_accuracy={correct/count:.4f}")

model.eval()
y_true, y_pred, y_prob = [], [], []
with torch.no_grad():
    for x, y in test_loader:
        probs = torch.softmax(model(x), dim=1)[:, 1]
        y_true.extend(y.tolist())
        y_pred.extend((probs >= 0.5).long().tolist())
        y_prob.extend(probs.tolist())

print("\nTest confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(y_true, y_pred))
print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"], zero_division=0))
print("Test ROC-AUC:", round(roc_auc_score(y_true, y_prob), 4))

Path("outputs").mkdir(exist_ok=True)
torch.save(model.state_dict(), "outputs/baseline_cnn.pth")
print("Saved model to outputs/baseline_cnn.pth")
