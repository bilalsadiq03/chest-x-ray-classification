"""
Evaluation metrics and figure generation for PneumoBackstop.

Computes metrics only from real data. Does not fill in missing VLM
assessments with fabricated values. Reports partial coverage explicitly.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# CNN metrics
# ---------------------------------------------------------------------------

def compute_cnn_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[float],
) -> dict[str, Any]:
    """
    Compute binary classification metrics from ground-truth and predictions.

    Parameters
    ----------
    y_true : list of int   (0=NORMAL, 1=PNEUMONIA)
    y_pred : list of int   hard predictions at threshold 0.5
    y_prob : list of float pneumonia probabilities from the model

    Returns a dict of scalar metrics plus the confusion matrix.
    """
    from sklearn.metrics import (
        confusion_matrix, classification_report, roc_auc_score, f1_score,
        precision_score, recall_score,
    )

    arr_true = np.asarray(y_true, dtype=int)
    arr_pred = np.asarray(y_pred, dtype=int)
    arr_prob = np.asarray(y_prob, dtype=float)

    cm      = confusion_matrix(arr_true, arr_pred, labels=[0, 1]).tolist()
    report  = classification_report(
        arr_true, arr_pred,
        labels=[0, 1],
        target_names=["NORMAL", "PNEUMONIA"],
        output_dict=True,
        zero_division=0,
    )

    tn, fp, fn, tp = (
        cm[0][0], cm[0][1],
        cm[1][0], cm[1][1],
    )

    return {
        "test_samples":    len(arr_true),
        "test_accuracy":   float(np.mean(arr_true == arr_pred)),
        "test_roc_auc":    float(roc_auc_score(arr_true, arr_prob)),
        "precision_normal":     float(precision_score(arr_true, arr_pred, pos_label=0, zero_division=0)),
        "recall_normal":        float(recall_score(arr_true, arr_pred, pos_label=0, zero_division=0)),
        "f1_normal":            float(f1_score(arr_true, arr_pred, pos_label=0, zero_division=0)),
        "precision_pneumonia":  float(precision_score(arr_true, arr_pred, pos_label=1, zero_division=0)),
        "recall_pneumonia":     float(recall_score(arr_true, arr_pred, pos_label=1, zero_division=0)),
        "f1_pneumonia":         float(f1_score(arr_true, arr_pred, pos_label=1, zero_division=0)),
        "true_negatives":  int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives":  int(tp),
        "confusion_matrix": cm,
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# VLM coverage metrics
# ---------------------------------------------------------------------------

def compute_vlm_metrics(
    predictions_path: str | Path,
    vlm_path: str | Path,
) -> dict[str, Any]:
    """
    Compute second-look metrics from the predictions CSV and VLM assessments CSV.

    Only counts *successful* VLM assessments (no error, valid assessment field).
    All metrics are constrained to the assessed subset; combined-system metrics
    are NOT computed because VLM coverage is too small for meaningful statistics.
    """
    import csv

    predictions_path = Path(predictions_path)
    vlm_path         = Path(vlm_path)

    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    if not vlm_path.exists():
        raise FileNotFoundError(vlm_path)

    with predictions_path.open(newline="", encoding="utf-8-sig") as f:
        baseline = {r["path"]: r for r in csv.DictReader(f)}

    with vlm_path.open(newline="", encoding="utf-8-sig") as f:
        vlm_rows = list(csv.DictReader(f))

    VALID = {"PNEUMONIA_SUGGESTIVE", "NOT_PNEUMONIA_SUGGESTIVE", "INDETERMINATE"}

    # Only count rows with a valid assessment and no error
    successful = [
        r for r in vlm_rows
        if not r.get("error", "").strip()
        and r.get("assessment", "").strip() in VALID
        and r.get("path") in baseline
    ]
    failed_or_missing = len(vlm_rows) - len(successful)

    counts = Counter(r["assessment"] for r in successful)

    # CNN-predicted NORMAL cases selected for second look
    cnn_normal_paths = {
        p for p, r in baseline.items() if r["predicted_label"] == "NORMAL"
    }

    # CNN false negatives
    all_fn = [
        r for r in baseline.values()
        if r["true_label"] == "PNEUMONIA" and r["predicted_label"] == "NORMAL"
    ]
    assessed_paths = {r["path"] for r in successful}

    assessed_fn = [r for r in all_fn if r["path"] in assessed_paths]
    flagged_fn  = [
        r for r in successful
        if r["assessment"] == "PNEUMONIA_SUGGESTIVE"
        and baseline[r["path"]]["true_label"] == "PNEUMONIA"
    ]
    flagged_true_normal = [
        r for r in successful
        if r["assessment"] == "PNEUMONIA_SUGGESTIVE"
        and baseline[r["path"]]["true_label"] == "NORMAL"
    ]

    return {
        "total_test_images":          len(baseline),
        "cnn_predicted_normal":       len(cnn_normal_paths),
        "vlm_rows_total":             len(vlm_rows),
        "vlm_successful":             len(successful),
        "vlm_failed_or_unusable":     failed_or_missing,
        "vlm_coverage_pct":           (
            100.0 * len(successful) / len(cnn_normal_paths)
            if cnn_normal_paths else 0.0
        ),
        "vlm_assessment_counts":      dict(counts),
        "all_cnn_false_negatives":    len(all_fn),
        "fn_assessed_by_vlm":         len(assessed_fn),
        "fn_flagged_pneumonia_suggestive": len(flagged_fn),
        "flagged_but_true_normal":    len(flagged_true_normal),
        "note": (
            "Metrics cover only the successfully assessed subset. "
            "VLM coverage was limited by API quota exhaustion. "
            "Combined-system metrics are not computed due to insufficient coverage."
        ),
    }


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: list[list[int]],
    output_path: str | Path,
    *,
    n_test: int | None = None,
) -> None:
    """Render and save the CNN confusion matrix heatmap."""
    arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap="Blues")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["NORMAL", "PNEUMONIA"],
        yticklabels=["NORMAL", "PNEUMONIA"],
        xlabel="Predicted label",
        ylabel="True label",
        title=(
            "CNN Test Confusion Matrix"
            + (f"  (n={n_test})" if n_test else "")
        ),
    )
    for i in range(2):
        for j in range(2):
            color = "white" if arr[i, j] > arr.max() / 2 else "black"
            ax.text(j, i, str(arr[i, j]), ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Number of images")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)


def plot_probability_distribution(
    predictions_path: str | Path,
    output_path: str | Path,
) -> None:
    """Plot CNN pneumonia probability distribution by true class."""
    import csv

    with Path(predictions_path).open(newline="", encoding="utf-8-sig") as f:
        predictions = list(csv.DictReader(f))

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, name, color in [
        ("NORMAL",    "True NORMAL",    "steelblue"),
        ("PNEUMONIA", "True PNEUMONIA", "tomato"),
    ]:
        values = [
            float(r["pneumonia_probability"])
            for r in predictions
            if r["true_label"] == label
        ]
        ax.hist(values, bins=20, alpha=0.6, label=name, color=color)

    ax.axvline(0.5, linestyle="--", color="black", label="Decision threshold = 0.5")
    ax.set(
        xlabel="CNN predicted probability of pneumonia",
        ylabel="Number of images",
        title=f"CNN Probability Distribution — Held-Out Test Set (n={len(predictions)})",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)


def plot_vlm_assessments(
    vlm_path: str | Path,
    output_path: str | Path,
) -> None:
    """Bar chart of VLM assessment categories for successfully assessed images."""
    import csv

    VALID = {"PNEUMONIA_SUGGESTIVE", "NOT_PNEUMONIA_SUGGESTIVE", "INDETERMINATE"}
    with Path(vlm_path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    successful = [
        r for r in rows
        if not r.get("error", "").strip()
        and r.get("assessment", "").strip() in VALID
    ]
    counts = Counter(r["assessment"] for r in successful)

    labels = [
        "PNEUMONIA_\nSUGGESTIVE",
        "NOT_PNEUMONIA_\nSUGGESTIVE",
        "INDETERMINATE",
    ]
    keys = [
        "PNEUMONIA_SUGGESTIVE",
        "NOT_PNEUMONIA_SUGGESTIVE",
        "INDETERMINATE",
    ]
    colors = ["tomato", "steelblue", "grey"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, [counts[k] for k in keys], color=colors, width=0.5)
    ax.set(
        ylabel="Number of assessments",
        title=(
            f"Gemini Second-Look Assessments\n"
            f"(n={len(successful)} successfully assessed "
            f"of {len(rows)} submitted)"
        ),
    )
    for i, (k, label) in enumerate(zip(keys, labels)):
        ax.text(i, counts[k] + 0.1, str(counts[k]), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)
