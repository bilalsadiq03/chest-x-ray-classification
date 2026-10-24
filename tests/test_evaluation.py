"""
Unit tests for evaluation metrics.
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import compute_cnn_metrics, compute_vlm_metrics


class TestComputeCnnMetrics:

    def _perfect_predictions(self):
        y_true = [0, 0, 1, 1, 1]
        y_pred = [0, 0, 1, 1, 1]
        y_prob = [0.1, 0.2, 0.8, 0.9, 0.95]
        return y_true, y_pred, y_prob

    def test_perfect_accuracy(self):
        y_true, y_pred, y_prob = self._perfect_predictions()
        m = compute_cnn_metrics(y_true, y_pred, y_prob)
        assert m["test_accuracy"] == pytest.approx(1.0)

    def test_confusion_matrix_shape(self):
        y_true, y_pred, y_prob = self._perfect_predictions()
        m = compute_cnn_metrics(y_true, y_pred, y_prob)
        assert len(m["confusion_matrix"]) == 2
        assert len(m["confusion_matrix"][0]) == 2

    def test_false_negatives_counted(self):
        # 1 FN: true PNEUMONIA (1) predicted NORMAL (0)
        y_true = [0, 0, 1, 1, 1]
        y_pred = [0, 0, 0, 1, 1]
        y_prob = [0.1, 0.2, 0.4, 0.8, 0.9]
        m = compute_cnn_metrics(y_true, y_pred, y_prob)
        assert m["false_negatives"] == 1
        assert m["false_positives"] == 0

    def test_test_samples_count(self):
        y_true = [0] * 10 + [1] * 10
        y_pred = [0] * 10 + [1] * 10
        y_prob = [0.1] * 10 + [0.9] * 10
        m = compute_cnn_metrics(y_true, y_pred, y_prob)
        assert m["test_samples"] == 20

    def test_roc_auc_perfect(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        y_prob = [0.1, 0.2, 0.8, 0.9]
        m = compute_cnn_metrics(y_true, y_pred, y_prob)
        assert m["test_roc_auc"] == pytest.approx(1.0)


class TestComputeVlmMetrics:

    def _write_csv(self, path: Path, rows: list[dict], fieldnames: list[str]):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_basic_coverage_metrics(self, tmp_path):
        # 4 test images: 2 predicted NORMAL, 2 PNEUMONIA
        preds = [
            {"path": "img1.jpg", "true_label": "NORMAL",    "predicted_label": "NORMAL",    "pneumonia_probability": "0.1", "correct": "1", "case_type": "TRUE_NEGATIVE"},
            {"path": "img2.jpg", "true_label": "PNEUMONIA",  "predicted_label": "NORMAL",    "pneumonia_probability": "0.4", "correct": "0", "case_type": "FALSE_NEGATIVE"},
            {"path": "img3.jpg", "true_label": "PNEUMONIA",  "predicted_label": "PNEUMONIA", "pneumonia_probability": "0.9", "correct": "1", "case_type": "TRUE_POSITIVE"},
            {"path": "img4.jpg", "true_label": "NORMAL",    "predicted_label": "PNEUMONIA", "pneumonia_probability": "0.7", "correct": "0", "case_type": "FALSE_POSITIVE"},
        ]
        pred_fields = ["path", "true_label", "predicted_label", "pneumonia_probability", "correct", "case_type"]

        vlm_rows = [
            {"path": "img1.jpg", "assessment": "NOT_PNEUMONIA_SUGGESTIVE", "error": "", "model": "gemini-flash", "findings": "", "limitations": "", "raw_response": ""},
            {"path": "img2.jpg", "assessment": "PNEUMONIA_SUGGESTIVE",     "error": "", "model": "gemini-flash", "findings": "", "limitations": "", "raw_response": ""},
        ]
        vlm_fields = ["path", "assessment", "error", "model", "findings", "limitations", "raw_response"]

        pred_path = tmp_path / "preds.csv"
        vlm_path  = tmp_path / "vlm.csv"
        self._write_csv(pred_path, preds, pred_fields)
        self._write_csv(vlm_path, vlm_rows, vlm_fields)

        m = compute_vlm_metrics(pred_path, vlm_path)
        assert m["total_test_images"] == 4
        assert m["cnn_predicted_normal"] == 2
        assert m["vlm_successful"] == 2
        assert m["all_cnn_false_negatives"] == 1
        assert m["fn_assessed_by_vlm"] == 1
        assert m["fn_flagged_pneumonia_suggestive"] == 1

    def test_errored_vlm_rows_not_counted(self, tmp_path):
        preds = [
            {"path": "img1.jpg", "true_label": "NORMAL", "predicted_label": "NORMAL",
             "pneumonia_probability": "0.1", "correct": "1", "case_type": "TRUE_NEGATIVE"},
        ]
        vlm_rows = [
            {"path": "img1.jpg", "assessment": "NOT_PNEUMONIA_SUGGESTIVE",
             "error": "API error occurred", "model": "x", "findings": "", "limitations": "", "raw_response": ""},
        ]
        pred_fields = ["path", "true_label", "predicted_label", "pneumonia_probability", "correct", "case_type"]
        vlm_fields  = ["path", "assessment", "error", "model", "findings", "limitations", "raw_response"]

        pred_path = tmp_path / "preds.csv"
        vlm_path  = tmp_path / "vlm.csv"
        self._write_csv(pred_path, preds, pred_fields)
        self._write_csv(vlm_path, vlm_rows, vlm_fields)

        m = compute_vlm_metrics(pred_path, vlm_path)
        assert m["vlm_successful"] == 0
        assert m["vlm_failed_or_unusable"] == 1

    def test_missing_files_raise(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compute_vlm_metrics(tmp_path / "nonexistent.csv", tmp_path / "also_missing.csv")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
