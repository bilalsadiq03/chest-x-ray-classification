import csv
import json
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("outputs")
FIGURES = OUT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

with (OUT / "baseline_results.json").open(encoding="utf-8") as f:
    baseline = json.load(f)

with (OUT / "test_predictions.csv").open(
    newline="", encoding="utf-8-sig"
) as f:
    predictions = list(csv.DictReader(f))

with (OUT / "gemini_second_look.csv").open(
    newline="", encoding="utf-8-sig"
) as f:
    vlm_rows = list(csv.DictReader(f))

# Figure 1: CNN confusion matrix.
cm = np.array(baseline["confusion_matrix"])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm)
ax.set(
    xticks=[0, 1],
    yticks=[0, 1],
    xticklabels=["NORMAL", "PNEUMONIA"],
    yticklabels=["NORMAL", "PNEUMONIA"],
    xlabel="Predicted label",
    ylabel="True label",
    title="CNN Test Confusion Matrix",
)
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")
fig.colorbar(im, ax=ax, label="Number of images")
fig.tight_layout()
fig.savefig(FIGURES / "cnn_confusion_matrix.png", dpi=160)
plt.close(fig)

# Figure 2: CNN pneumonia probability distribution by true class.
fig, ax = plt.subplots(figsize=(8, 5))
for label, name in [("NORMAL", "True NORMAL"), ("PNEUMONIA", "True PNEUMONIA")]:
    values = [
        float(r["pneumonia_probability"])
        for r in predictions
        if r["true_label"] == label
    ]
    ax.hist(values, bins=20, alpha=0.6, label=name)
ax.axvline(0.5, linestyle="--", label="Decision threshold = 0.5")
ax.set(
    xlabel="CNN predicted probability of pneumonia",
    ylabel="Number of images",
    title="CNN Probability Distribution on Held-Out Test Set",
)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "cnn_probability_distribution.png", dpi=160)
plt.close(fig)

# Figure 3: Gemini assessments among successful records only.
allowed = {
    "PNEUMONIA_SUGGESTIVE",
    "NOT_PNEUMONIA_SUGGESTIVE",
    "INDETERMINATE",
}
successful = [
    r for r in vlm_rows
    if not r.get("error", "").strip()
    and r.get("assessment", "").strip() in allowed
]
counts = Counter(r["assessment"] for r in successful)
labels = [
    "PNEUMONIA_SUGGESTIVE",
    "NOT_PNEUMONIA_SUGGESTIVE",
    "INDETERMINATE",
]
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(labels, [counts[x] for x in labels])
ax.set(
    ylabel="Number of assessments",
    title=f"Gemini Second-Look Assessments (n={len(successful)})",
)
ax.tick_params(axis="x", rotation=15)
fig.tight_layout()
fig.savefig(FIGURES / "gemini_assessment_counts.png", dpi=160)
plt.close(fig)

# Summary based on actual saved artifacts.
cm = baseline["confusion_matrix"]
misses = [
    r for r in predictions
    if r["true_label"] == "PNEUMONIA"
    and r["predicted_label"] == "NORMAL"
]
fn_assessed = [
    r for r in successful
    if r["path"] in {m["path"] for m in misses}
]
flagged_misses = [
    r for r in fn_assessed
    if r["assessment"] == "PNEUMONIA_SUGGESTIVE"
]

summary = {
    "test_samples": len(predictions),
    "cnn_test_accuracy": baseline["test_accuracy"],
    "cnn_test_roc_auc": baseline["test_roc_auc"],
    "cnn_confusion_matrix_rows_true_cols_pred": cm,
    "cnn_false_negatives": len(misses),
    "cnn_false_positives": sum(
        r["true_label"] == "NORMAL"
        and r["predicted_label"] == "PNEUMONIA"
        for r in predictions
    ),
    "gemini_successful_assessments_in_second_look_file": len(successful),
    "gemini_assessment_counts": dict(counts),
    "cnn_false_negatives_assessed_by_gemini": len(fn_assessed),
    "cnn_false_negatives_flagged_by_gemini": len(flagged_misses),
    "interpretation_warning": (
        "Gemini inference was quota-limited. Results describe only successfully "
        "assessed cases and do not establish clinical effectiveness."
    ),
    "figures": [
        "figures/cnn_confusion_matrix.png",
        "figures/cnn_probability_distribution.png",
        "figures/gemini_assessment_counts.png",
    ],
}

with (OUT / "experiment_summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, allow_nan=False)

print("Analysis figures generated:")
for path in sorted(FIGURES.glob("*.png")):
    print(" -", path)
print("Summary saved:", OUT / "experiment_summary.json")
print("CNN test accuracy:", round(baseline["test_accuracy"], 4))
print("CNN test ROC-AUC:", round(baseline["test_roc_auc"], 4))
print("CNN false negatives:", len(misses))
print("Successful Gemini assessments:", len(successful))
print("False negatives assessed by Gemini:", len(fn_assessed))
print("False negatives flagged by Gemini:", len(flagged_misses))
