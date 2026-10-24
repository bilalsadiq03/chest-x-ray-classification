"""
scripts/generate_analysis.py — Compute all metrics and regenerate all figures.

Usage (from project root):
    .venv\Scripts\python.exe scripts/generate_analysis.py

Reads:
  outputs/baseline_results.json
  outputs/test_predictions.csv
  outputs/gemini_second_look.csv

Writes:
  outputs/figures/cnn_confusion_matrix.png
  outputs/figures/cnn_probability_distribution.png
  outputs/figures/gemini_assessment_counts.png
  outputs/experiment_summary.json
  outputs/second_look_summary.txt
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import (
    compute_vlm_metrics,
    plot_confusion_matrix,
    plot_probability_distribution,
    plot_vlm_assessments,
)

OUT     = Path("outputs")
FIGURES = OUT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
Path("outputs/logs").mkdir(parents=True, exist_ok=True)

BASELINE_JSON = OUT / "baseline_results.json"
PREDS_CSV     = OUT / "test_predictions.csv"
VLM_CSV       = OUT / "gemini_second_look.csv"

# ── load baseline results ─────────────────────────────────────────────────────
if not BASELINE_JSON.exists():
    print(f"ERROR: {BASELINE_JSON} not found. Run scripts/recover_baseline.py first.", file=sys.stderr)
    sys.exit(1)

with BASELINE_JSON.open(encoding="utf-8") as f:
    baseline = json.load(f)

# ── Figure 1: Confusion matrix ────────────────────────────────────────────────
cm = baseline["confusion_matrix"]
n_test = baseline.get("classification_report", {}).get("accuracy") and len(baseline.get("test_labels", []))
plot_confusion_matrix(
    cm,
    FIGURES / "cnn_confusion_matrix.png",
    n_test=len(baseline.get("test_labels", [])) or None,
)
print("Saved:", FIGURES / "cnn_confusion_matrix.png")

# ── Figure 2: CNN probability distribution ────────────────────────────────────
if PREDS_CSV.exists():
    plot_probability_distribution(PREDS_CSV, FIGURES / "cnn_probability_distribution.png")
    print("Saved:", FIGURES / "cnn_probability_distribution.png")
else:
    print(f"WARNING: {PREDS_CSV} not found — skipping probability distribution figure.")

# ── Figure 3: VLM assessment bar chart ───────────────────────────────────────
if VLM_CSV.exists():
    plot_vlm_assessments(VLM_CSV, FIGURES / "gemini_assessment_counts.png")
    print("Saved:", FIGURES / "gemini_assessment_counts.png")
else:
    print(f"WARNING: {VLM_CSV} not found — skipping VLM assessment figure.")

# ── VLM metrics ────────────────────────────────────────────────────────────────
vlm_metrics: dict = {}
if PREDS_CSV.exists() and VLM_CSV.exists():
    vlm_metrics = compute_vlm_metrics(PREDS_CSV, VLM_CSV)

# ── Experiment summary JSON ────────────────────────────────────────────────────
summary = {
    "test_samples":            len(baseline.get("test_labels", [])),
    "cnn_test_accuracy":       baseline.get("test_accuracy"),
    "cnn_test_roc_auc":        baseline.get("test_roc_auc"),
    "cnn_confusion_matrix_rows_true_cols_pred": cm,
    "cnn_false_negatives":     cm[1][0] if cm else None,
    "cnn_false_positives":     cm[0][1] if cm else None,
    **vlm_metrics,
    "figures": [
        "figures/cnn_confusion_matrix.png",
        "figures/cnn_probability_distribution.png",
        "figures/gemini_assessment_counts.png",
    ],
}

with (OUT / "experiment_summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, allow_nan=False)
print("Saved:", OUT / "experiment_summary.json")

# ── Text summary ───────────────────────────────────────────────────────────────
lines = [
    "PNEUMOBACKSTOP — SECOND-LOOK EXPLORATORY SUMMARY",
    "=" * 52,
    f"Total held-out test images:          {summary.get('test_samples', 'N/A')}",
    f"CNN test accuracy:                   {baseline.get('test_accuracy', 0):.4f}",
    f"CNN test ROC-AUC:                    {baseline.get('test_roc_auc', 0):.4f}",
    f"CNN confusion matrix [[TN,FP],[FN,TP]]: {cm}",
    f"CNN false negatives:                 {cm[1][0] if cm else 'N/A'}",
    f"CNN false positives:                 {cm[0][1] if cm else 'N/A'}",
    "",
]
if vlm_metrics:
    counts = vlm_metrics.get("vlm_assessment_counts", {})
    lines += [
        f"CNN-predicted NORMAL (second-look candidates): {vlm_metrics.get('cnn_predicted_normal', 'N/A')}",
        f"Successful VLM assessments:          {vlm_metrics.get('vlm_successful', 'N/A')}",
        f"VLM coverage (%):                    {vlm_metrics.get('vlm_coverage_pct', 0):.1f}%",
        f"VLM failed/unusable:                 {vlm_metrics.get('vlm_failed_or_unusable', 'N/A')}",
        "",
        "VLM assessment breakdown:",
        f"  PNEUMONIA_SUGGESTIVE:              {counts.get('PNEUMONIA_SUGGESTIVE', 0)}",
        f"  NOT_PNEUMONIA_SUGGESTIVE:          {counts.get('NOT_PNEUMONIA_SUGGESTIVE', 0)}",
        f"  INDETERMINATE:                     {counts.get('INDETERMINATE', 0)}",
        "",
        f"CNN false negatives assessed by VLM: {vlm_metrics.get('fn_assessed_by_vlm', 0)}",
        f"CNN false negatives flagged (PS):    {vlm_metrics.get('fn_flagged_pneumonia_suggestive', 0)}",
        "",
        vlm_metrics.get("note", ""),
    ]
else:
    lines.append("VLM results file not found. Run scripts/run_second_look.py.")

report = "\n".join(lines)
(OUT / "second_look_summary.txt").write_text(report, encoding="utf-8")
print("Saved:", OUT / "second_look_summary.txt")
print()
print(report)
