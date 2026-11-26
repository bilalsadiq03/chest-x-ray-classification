import argparse, csv, json, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "outputs" / "splits"
CHECKPOINTS = ROOT / "checkpoints"
OUT = ROOT / "outputs" / "transfer_learning"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

class XrayDataset(Dataset):
    def __init__(self, manifest, training=False):
        self.df = pd.read_csv(manifest)
        self.training = training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        if not {"path", "label"}.issubset(self.df.columns):
            raise ValueError("Manifest must contain path and label columns")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(row["path"]) as im:
            image = im.convert("RGB")
        if self.training:
            if random.random() < 0.5:
                image = transforms.functional.hflip(image)
            image = transforms.functional.rotate(
                image, random.uniform(-5, 5),
                interpolation=transforms.InterpolationMode.BILINEAR
            )
        label = 0 if str(row["label"]).upper() == "NORMAL" else 1
        return self.transform(image), torch.tensor(label), str(row["path"])

def make_model(name):
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for p in m.parameters(): p.requires_grad = False
        for p in m.layer4.parameters(): p.requires_grad = True
        m.fc = nn.Linear(m.fc.in_features, 2)
    else:
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        for p in m.parameters(): p.requires_grad = False
        for p in m.features.denseblock4.parameters(): p.requires_grad = True
        for p in m.features.norm5.parameters(): p.requires_grad = True
        m.classifier = nn.Linear(m.classifier.in_features, 2)
    return m

def metrics(y, p):
    pred = (np.asarray(p) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall_pneumonia": float(recall_score(y, pred, zero_division=0)),
        "f1_score": float(f1_score(y, pred, zero_division=0)),
        "specificity_normal": float(tn / (tn + fp)) if tn + fp else 0.0,
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]
    }

def train_one(name, epochs, batch_size, device):
    print(f"\\n===== {name.upper()} =====", flush=True)
    train_ds = XrayDataset(SPLITS / "train.csv", True)
    val_ds = XrayDataset(SPLITS / "validation.csv", False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    labels = train_ds.df["label"].str.upper()
    n0, n1 = int((labels == "NORMAL").sum()), int((labels == "PNEUMONIA").sum())
    weights = torch.tensor([(n0+n1)/(2*n0), (n0+n1)/(2*n1)], device=device)
    model = make_model(name).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=1e-4, weight_decay=1e-4)
    best_loss = float("inf")
    best_path = CHECKPOINTS / f"best_{name}.pth"
    history = []

    for epoch in range(1, epochs+1):
        model.train()
        total_loss, count = 0.0, 0
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            count += len(y)
        train_loss = total_loss / count

        model.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for x, y, _ in val_dl:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * len(y)
                val_count += len(y)
        val_loss /= val_count
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        print(f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f}, val_loss={val_loss:.4f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model_name": name, "model_state_dict": model.state_dict(),
                        "image_size": 224, "class_to_id": {"NORMAL": 0, "PNEUMONIA": 1}},
                       best_path)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    paths, ys, ps = [], [], []
    with torch.no_grad():
        for x, y, batch_paths in val_dl:
            prob = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
            paths.extend(batch_paths); ys.extend(y.numpy().tolist()); ps.extend(prob.tolist())

    result = metrics(np.array(ys), np.array(ps))
    result.update({"model": name, "best_validation_loss": best_loss})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}_validation_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["path", "true_label", "p_pneumonia", "predicted_label"])
        for path, y, p in zip(paths, ys, ps): w.writerow([path, y, p, int(p >= .5)])
    (OUT / f"{name}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (OUT / f"{name}_validation_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Validation metrics:", json.dumps(result, indent=2), flush=True)
    print("Saved:", best_path, flush=True)
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    random.seed(42); 
    np.random.seed(42); 
    torch.manual_seed(42)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    CHECKPOINTS.mkdir(exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| epochs:", args.epochs, "| batch size:", args.batch_size)
    results = [train_one(n, args.epochs, args.batch_size, device)
               for n in ("resnet18", "densenet121")]
    (OUT / "transfer_learning_comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\\nBoth models finished. Results:", OUT)

if __name__ == "__main__":
    main()
