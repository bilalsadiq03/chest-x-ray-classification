import csv, json
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
OUT = ROOT / "outputs" / "final_model_evaluation"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

class XrayDataset(Dataset):
    def __init__(self, manifest):
        self.df = pd.read_csv(manifest)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(row["path"]) as im: image = im.convert("RGB")
        label = 0 if str(row["label"]).upper() == "NORMAL" else 1
        return self.transform(image), torch.tensor(label), str(row["path"])

def load_model(name, device):
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    else:
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    ckpt = torch.load(CHECKPOINTS / f"best_{name}.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()

def predict(model, loader, device):
    paths, ys, ps = [], [], []
    with torch.no_grad():
        for x, y, batch_paths in loader:
            p = torch.softmax(model(x.to(device)), dim=1)[:, 1]
            paths.extend(batch_paths); ys.extend(y.numpy().tolist()); ps.extend(p.cpu().numpy().tolist())
    return paths, np.asarray(ys), np.asarray(ps)

def score(y, p):
    pred = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "pneumonia_recall": float(recall_score(y, pred, zero_division=0)),
        "f1_score": float(f1_score(y, pred, zero_division=0)),
        "normal_specificity": float(tn/(tn+fp)) if tn+fp else 0.0,
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }

def save_csv(file, paths, ys, ps):
    with open(file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["path", "true_label_id", "p_pneumonia", "predicted_label_id"])
        for path, y, p in zip(paths, ys, ps): w.writerow([path, int(y), float(p), int(p >= .5)])

def main():
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    models_loaded = {n: load_model(n, device) for n in ("resnet18", "densenet121")}
    results = {}
    for split in ("validation", "test"):
        print(f"\\n===== {split.upper()} =====", flush=True)
        ds = XrayDataset(SPLITS / f"{split}.csv")
        dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
        probs_by_model = {}
        common_paths = common_y = None
        for name, model in models_loaded.items():
            paths, ys, ps = predict(model, dl, device)
            if common_paths is None: common_paths, common_y = paths, ys
            elif paths != common_paths or not np.array_equal(ys, common_y):
                raise RuntimeError("Prediction order mismatch between models")
            probs_by_model[name] = ps
            result = score(ys, ps); results[f"{split}_{name}"] = result
            save_csv(OUT / f"{split}_{name}_predictions.csv", paths, ys, ps)
            print(f"\\n{name.upper()}:\\n" + json.dumps(result, indent=2), flush=True)
        ensemble = (probs_by_model["resnet18"] + probs_by_model["densenet121"]) / 2
        result = score(common_y, ensemble)
        results[f"{split}_soft_voting_ensemble"] = result
        save_csv(OUT / f"{split}_ensemble_predictions.csv", common_paths, common_y, ensemble)
        print("\\nSOFT-VOTING ENSEMBLE:\\n" + json.dumps(result, indent=2), flush=True)
    (OUT / "all_model_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\\nDone. Results saved to:", OUT)
    print("Ensemble = arithmetic mean of ResNet18 and DenseNet121 pneumonia probabilities; threshold = 0.50.")

if __name__ == "__main__": main()
