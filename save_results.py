import json
from pathlib import Path

results = {
    "model": "Custom CNN trained from scratch",
    "best_validation_epoch": 4,
    "best_validation_roc_auc": 0.9890,
    "test_loss": 0.5557,
    "test_accuracy": 0.7718,
    "test_roc_auc": 0.9375,
    "confusion_matrix": [[95, 136], [5, 382]],
    "class_order": ["NORMAL", "PNEUMONIA"],
    "normal": {
        "precision": 0.95,
        "recall": 0.41,
        "f1_score": 0.57,
        "support": 231
    },
    "pneumonia": {
        "precision": 0.74,
        "recall": 0.99,
        "f1_score": 0.84,
        "support": 387
    },
    "note": "Metrics rounded to match the displayed terminal output."
}

Path("outputs").mkdir(exist_ok=True)
with open("outputs/baseline_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Saved outputs/baseline_results.json")
