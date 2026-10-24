# PROJECT_GUIDE.md — PneumoBackstop File Structure

This guide explains every important file and directory in plain English.

---

## Root-Level Files

| File | Purpose |
|------|---------|
| `README.md` | Setup, results, architecture, and reproduction commands |
| `PROJECT_GUIDE.md` | This file — explains the project structure |
| `requirements.txt` | Pinned Python dependencies (not including transitive packages) |
| `.gitignore` | Excludes `.env`, dataset, checkpoints, and generated outputs |
| `.env` | **Secret** — contains `GEMINI_API_KEY`. Never committed. |
| `configs/experiment.toml` | Non-secret configuration (paths, hyperparameters, VLM model name) |

---

## `src/` — Reusable Library Code

These modules are imported by scripts and tests. They contain no side effects when imported.

```
src/
  __init__.py
  data/
    __init__.py
    dataset.py          # ChestXrayDataset: loads images from CSV manifests,
                        # applies ImageNet normalization, optional augmentation
  models/
    __init__.py
    cnn.py              # SmallCNN architecture + build_model() helper
  vlm/
    __init__.py
    client.py           # Gemini API client: assess_image(), parse_vlm_response(),
                        # AssessmentStatus enum, quota-stop logic, no secret logging
  evaluation/
    __init__.py
    metrics.py          # compute_cnn_metrics(), compute_vlm_metrics(),
                        # plot_confusion_matrix(), plot_probability_distribution(),
                        # plot_vlm_assessments()
```

---

## `scripts/` — Command-Line Entry Points

Each script is meant to be run from the project root with the venv active.

| Script | What it does |
|--------|-------------|
| `validate_dataset.py` | Checks all manifest CSVs: file existence, label validity, hash overlap |
| `recover_baseline.py` | Loads CNN checkpoint, runs test-set inference, writes `outputs/baseline_results.json` and `outputs/test_predictions.csv` |
| `run_second_look.py` | Sends CNN-predicted-NORMAL images to Gemini API; resumes safely on re-run; stops on quota errors |
| `generate_analysis.py` | Reads saved artifacts, computes all metrics, regenerates all figures |

---

## Root-Level Scripts (Legacy — Preserved)

These older scripts at the root were the original implementation. They are kept as a reference and for auditability; the `scripts/` versions are the maintained equivalents.

| Script | Status | Notes |
|--------|--------|-------|
| `baseline.py` | Legacy | Original quick-prototype CNN; uses grayscale 96×96 images |
| `train_baseline.py` | **Active** | Full training script; saves `checkpoints/best_cnn.pth` |
| `rebuild_preprocessing.py` | **Active** | Scans raw data, deduplicates, writes split CSVs to `outputs/splits/` |
| `recover_baseline_results.py` | Superseded | Replaced by `scripts/recover_baseline.py` |
| `run_gemini_vlm.py` | Legacy | First VLM attempt; runs on full test set without CNN filtering |
| `run_gemini_second_look.py` | Superseded | Replaced by `scripts/run_second_look.py` |
| `analyze_second_look.py` | Superseded | Replaced by `scripts/generate_analysis.py` |
| `generate_analysis.py` (root) | Superseded | Replaced by `scripts/generate_analysis.py` |
| `save_results.py` | Legacy | Hardcoded rounded metrics; kept for reference only |
| `test_gemini_vlm.py` | Utility | One-shot API connectivity test |

---

## `tests/` — Unit Tests

All tests run locally with no API calls needed.

| File | Covers |
|------|--------|
| `test_dataset.py` | CSV loading, label IDs, tensor shape, normalization, error handling |
| `test_model.py` | CNN output shape, softmax probabilities, checkpoint round-trip |
| `test_vlm_parsing.py` | All valid assessments, malformed/empty responses, status enum values |
| `test_evaluation.py` | CNN metrics, VLM coverage counts, error-row exclusion |

Run with: `.venv\Scripts\python.exe -m pytest tests/ -v`

---

## `data/` — Dataset (Not Committed)

```
data/
  raw/
    chest_xray/         # Original dataset (gitignored)
      train/ val/ test/ — each with NORMAL/ and PNEUMONIA/ subfolders
  metadata/             # Reserved for future use
  processed/            # Reserved for preprocessed tensors (gitignored)
```

---

## `checkpoints/` — Model Weights (Not Committed)

```
checkpoints/
  best_cnn.pth          # Best validation-AUC checkpoint from train_baseline.py
                        # 330 KB — gitignored
```

---

## `outputs/` — Generated Artifacts (Partially Committed)

```
outputs/
  splits/               # Train/val/test CSV manifests (committed — small, reproducibility-critical)
    train.csv
    validation.csv
    test.csv
  eda/                  # Duplicate analysis manifests (gitignored — large)
    dataset_manifest_original.csv
    dataset_manifest_clean.csv
    eda_summary.json
  baseline_results.json # Full CNN test metrics (committed — 26 KB)
  test_predictions.csv  # Per-image CNN predictions (committed — 86 KB)
  training_history.json # Epoch-by-epoch training metrics
  gemini_second_look.csv      # VLM assessments for CNN-normal images
  gemini_vlm_assessments.csv  # First VLM run (full test set attempt)
  experiment_summary.json     # Combined metrics JSON
  second_look_summary.txt     # Human-readable report
  figures/              # PNG plots (gitignored — regeneratable)
    cnn_confusion_matrix.png
    cnn_probability_distribution.png
    gemini_assessment_counts.png
  logs/                 # Script run logs (gitignored)
```

---

## `reports/` — Final Academic Report

```
reports/
  final_report.md       # Full academic write-up: methodology, results, limitations
```

---

## `notebooks/` — Exploratory Notebooks

```
notebooks/
  02_preprocessing.ipynb   # Early EDA notebook (preserved for reference)
```
