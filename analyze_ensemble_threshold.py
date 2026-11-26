import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent
PRED_DIR = ROOT / "outputs" / "final_model_evaluation"
OUT_DIR = ROOT / "outputs" / "threshold_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def calculate(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "pneumonia_recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1_score": float(f1_score(y, pred, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }

def main():
    val = pd.read_csv(PRED_DIR / "validation_ensemble_predictions.csv")
    test = pd.read_csv(PRED_DIR / "test_ensemble_predictions.csv")

    y_val = val["true_label_id"].to_numpy(dtype=int)
    p_val = val["p_pneumonia"].to_numpy(dtype=float)
    y_test = test["true_label_id"].to_numpy(dtype=int)
    p_test = test["p_pneumonia"].to_numpy(dtype=float)

    # Candidate thresholds are chosen on validation data only.
    thresholds = np.round(np.arange(0.10, 0.951, 0.01), 2)
    val_results = [calculate(y_val, p_val, t) for t in thresholds]

    # Example operating point: maximize validation F1. In case of ties,
    # choose the higher threshold to prefer fewer positive predictions.
    best = sorted(val_results, key=lambda r: (r["f1_score"], r["threshold"]), reverse=True)[0]
    chosen_threshold = best["threshold"]

    # Apply the frozen validation-selected threshold to the held-out test set.
    test_at_selected = calculate(y_test, p_test, chosen_threshold)
    test_at_default = calculate(y_test, p_test, 0.50)

    summary = {
        "selection_method": "Maximum validation F1; tie-break toward higher threshold",
        "selected_threshold_from_validation": chosen_threshold,
        "validation_metrics_at_selected_threshold": best,
        "test_metrics_at_default_threshold_0.50": test_at_default,
        "test_metrics_at_validation_selected_threshold": test_at_selected,
        "validation_roc_auc": float(roc_auc_score(y_val, p_val)),
        "validation_pr_auc": float(average_precision_score(y_val, p_val)),
        "test_roc_auc": float(roc_auc_score(y_test, p_test)),
        "test_pr_auc": float(average_precision_score(y_test, p_test))
    }

    pd.DataFrame(val_results).to_csv(OUT_DIR / "validation_threshold_sweep.csv", index=False)
    (OUT_DIR / "threshold_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Threshold selected using VALIDATION only:", chosen_threshold)
    print("\nValidation metrics at selected threshold:")
    print(json.dumps(best, indent=2))
    print("\nHeld-out TEST at default threshold 0.50:")
    print(json.dumps(test_at_default, indent=2))
    print("\nHeld-out TEST at frozen validation-selected threshold:")
    print(json.dumps(test_at_selected, indent=2))
    print("\nSaved threshold sweep and summary to:", OUT_DIR)

if __name__ == "__main__":
    main()
