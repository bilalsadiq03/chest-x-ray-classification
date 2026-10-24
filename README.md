# PneumoBackstop

**A Generative AI-Based Second-Look System for Pneumonia Detection in Chest X-Rays**  
*CSET343 — AI in Healthcare | Academic Prototype*

> **Disclaimer**: This is a research prototype developed for a university course. It is not a clinically validated medical device and must not be used for diagnostic purposes.

---

## Overview

PneumoBackstop investigates whether a generative vision-language model (VLM) can complement a lightweight CNN classifier by independently assessing chest X-rays that the CNN classifies as NORMAL — potentially identifying cases that deserve additional human review.

The pipeline:
1. Train a small CNN from scratch on publicly available chest X-ray images
2. Evaluate the CNN on a held-out test set
3. Select CNN-predicted NORMAL cases for a "second look"
4. Send those images to the Gemini VLM API (without ground-truth labels)
5. Compare VLM assessments against ground truth offline

---

## Results Summary

| Metric | Value |
|--------|-------|
| Test images | 618 |
| CNN test accuracy | 0.7718 |
| CNN test ROC-AUC | 0.9375 |
| CNN confusion matrix (rows=true, cols=pred) | [[95, 136], [5, 382]] |
| CNN false negatives | 5 |
| CNN false positives | 136 |
| CNN-normal cases selected for second look | 100 |
| Successful VLM assessments | 18 / 100 (18%) |
| VLM: PNEUMONIA_SUGGESTIVE | 0 |
| VLM: NOT_PNEUMONIA_SUGGESTIVE | 14 |
| VLM: INDETERMINATE | 4 |
| CNN false negatives assessed by VLM | 0 / 5 |
| CNN false negatives flagged by VLM | 0 |

> **VLM coverage gap**: The Gemini API hit its free-tier quota limit after 18 of 100 requests. None of the 5 CNN false-negative images received a successful assessment. Combined-system metrics cannot be meaningfully calculated from an 18% coverage subset.

---

## Setup

### Requirements

- Python 3.10.x (tested with 3.10.8)
- Windows (PowerShell) or Linux/macOS

### Install

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install PyTorch (CPU-only, Windows)
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining dependencies
pip install -r requirements.txt
```

### Configure API Key

```powershell
# Create .env in the project root (already gitignored):
# GEMINI_API_KEY="your_key_here"

# Or set it in the shell:
$env:GEMINI_API_KEY = "your_key_here"
```

---

## Data

Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney, licensed CC BY 4.0.

Place the dataset at `data/raw/chest_xray/` with the structure:

```
data/raw/chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

**Dataset statistics (after SHA-256 duplicate removal):**

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,078 | 3,086 | 4,164 |
| Validation | 269 | 773 | 1,042 |
| Test | 234 | 390 | 618 (original test split) |
| **Total** | **1,581** | **4,249** | **5,824** |

---

## Exact Reproduction Commands

Run these in order from the project root with the venv active:

```powershell
# Step 1: Validate dataset manifests
.venv\Scripts\python.exe scripts\validate_dataset.py

# Step 2: Rebuild preprocessing manifests (if needed)
.venv\Scripts\python.exe rebuild_preprocessing.py

# Step 3: Train the CNN (skip if checkpoint exists)
.venv\Scripts\python.exe train_baseline.py

# Step 4: Re-run CNN inference on the test set
.venv\Scripts\python.exe scripts\recover_baseline.py

# Step 5: Run the VLM second-look (requires GEMINI_API_KEY)
.venv\Scripts\python.exe scripts\run_second_look.py

# Step 6: Generate all figures and the experiment summary
.venv\Scripts\python.exe scripts\generate_analysis.py

# Step 7: Run unit tests
.venv\Scripts\python.exe -m pytest tests/ -v
```

---

## Architecture

```
CNN (SmallCNN, trained from scratch)
  Input: 224×224 RGB chest X-ray
  Conv(3→16) + BN + ReLU + MaxPool
  Conv(16→32) + BN + ReLU + MaxPool
  Conv(32→64) + BN + ReLU + MaxPool
  Conv(64→96) + BN + ReLU + AdaptiveAvgPool
  Dropout(0.3) → Linear(96 → 2)
  Output: logits for [NORMAL, PNEUMONIA]

VLM (Gemini Flash, accessed via API)
  Input: chest X-ray image bytes (no ground-truth labels)
  Prompt: structured assessment request
  Output: PNEUMONIA_SUGGESTIVE | NOT_PNEUMONIA_SUGGESTIVE | INDETERMINATE
```

---

## Limitations

1. **VLM quota gap**: Only 18/100 second-look images were assessed before the API quota was exhausted. None of the 5 false negatives were reached. Combined-system conclusions cannot be drawn from this incomplete run.
2. **Image-level split**: Splits are by image, not patient. If multiple images per patient exist, split independence is not guaranteed.
3. **Cross-split duplicates**: SHA-256 checking found 0 cross-split exact duplicates, but visually-similar re-encodings would pass this check.
4. **CNN not pretrained**: The CNN was trained from scratch on a consumer CPU. It has no pretrained medical radiology knowledge.
5. **VLM not clinically validated**: Gemini Flash is a general-purpose model, not a radiology-specific validated tool.
6. **Class imbalance**: PNEUMONIA cases outnumber NORMAL ~2.7:1, leading to high pneumonia recall (0.99) but very low normal recall (0.41).

---

## File Structure

See [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) for a detailed description of every directory and script.
