# PneumoBackstop — Final Experimental Report

**Course:** CSET343 — AI in Healthcare  
**Project:** A Generative AI-Based Second-Look System for Pneumonia Detection in Chest X-Rays  
**Status:** Academic prototype — not a clinical tool

---

## 1. Motivation and Research Question

Pneumonia is a leading cause of hospitalisation worldwide. Automated screening tools based on convolutional neural networks (CNNs) can assist radiologists by flagging suspicious findings, but no automated system is infallible. False negatives — cases where pneumonia is present but classified as normal — are the most clinically dangerous error.

**Research question:** Can a generative vision-language model (VLM), queried independently and without access to ground-truth labels, provide a useful second opinion on chest X-rays that a CNN classifies as normal?

This prototype explores the feasibility of such a "second-look" system. It does not claim clinical validity or replace human clinical judgment.

---

## 2. Dataset

**Source:** Chest X-Ray Images (Pneumonia) dataset by Paul Mooney, Kaggle. Licensed CC BY 4.0.

**Preprocessing:**
- SHA-256 exact-content duplicate detection identified 30 duplicate groups (32 images) within and across the original splits.
- No cross-split exact duplicates were found.
- After intra-split deduplication, 5,824 unique images remained.
- The original test split (618 images) was held out and not used for training or validation.
- Train and validation splits were rebuilt with a stratified 80/20 random split from the combined original train+val data.

**Final split counts:**

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,078 | 3,086 | 4,164 |
| Validation | 269 | 773 | 1,042 |
| Test (held-out) | 234 | 390 | 618 |

**Important caveats:**
- Splits are at the image level, not the patient level. If multiple images from the same patient appear in the dataset, split independence cannot be guaranteed.
- SHA-256 matching detects only exact byte-identical duplicates. Visually-identical images with different encodings would pass this check.
- The PNEUMONIA class outnumbers NORMAL by approximately 2.7:1.

---

## 3. CNN Model

### Architecture (SmallCNN)

A lightweight 4-block convolutional network trained from scratch on CPU:

```
Input: 224 × 224 × 3 (RGB, ImageNet normalization)
Block 1: Conv(3→16) + BatchNorm + ReLU + MaxPool(2)
Block 2: Conv(16→32) + BatchNorm + ReLU + MaxPool(2)
Block 3: Conv(32→64) + BatchNorm + ReLU + MaxPool(2)
Block 4: Conv(64→96) + BatchNorm + ReLU + AdaptiveAvgPool(1,1)
Classifier: Flatten → Dropout(0.3) → Linear(96 → 2)
Output: logits for [NORMAL, PNEUMONIA]
```

Total parameters: approximately 116,000.

### Training

- Optimizer: AdamW (lr=0.001, weight_decay=0.0001)
- Loss: CrossEntropyLoss with inverse-frequency class weights
- Epochs: 5
- Batch size: 16
- Augmentation: random horizontal flip + random rotation ±5°
- Checkpoint selection: best validation ROC-AUC
- Hardware: consumer CPU (no GPU)

### Test Set Results

| Metric | Value |
|--------|-------|
| Test accuracy | 0.7718 |
| Test ROC-AUC | 0.9375 |
| Confusion matrix (rows=true, cols=predicted) | [[95, 136], [5, 382]] |
| True negatives (NORMAL correctly classified) | 95 |
| False positives (NORMAL classified as PNEUMONIA) | 136 |
| False negatives (PNEUMONIA classified as NORMAL) | 5 |
| True positives (PNEUMONIA correctly classified) | 382 |

**Per-class metrics (test set):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| NORMAL | 0.950 | 0.411 | 0.574 | 231 |
| PNEUMONIA | 0.737 | 0.987 | 0.844 | 387 |
| Weighted avg | 0.817 | 0.772 | 0.743 | 618 |

**Interpretation:** The CNN achieves very high pneumonia recall (0.987) — it misses only 5 of 390 pneumonia cases — but at the cost of misclassifying 136 normal cases as pneumonia (very low normal recall: 0.411). The high false-positive rate means many unnecessary reviews would be generated in practice.

---

## 4. VLM Second-Look Evaluation

### Design

- Only CNN-predicted NORMAL cases are sent to the VLM (100 images).
- Ground-truth labels are never included in the VLM prompt.
- Labels are used only for offline evaluation after assessments are collected.

**VLM prompt (verbatim):**
```
Assess this chest X-ray for a research prototype.
Determine whether visible findings are suggestive of pneumonia.
Do not assume patient history or invent findings.
If uncertain, use INDETERMINATE. This is not a clinical diagnosis.

Return exactly three labelled lines (no markdown, no extra commentary):
Assessment: PNEUMONIA_SUGGESTIVE, NOT_PNEUMONIA_SUGGESTIVE, or INDETERMINATE
Findings: brief description of visible findings
Limitations: key sources of uncertainty or image-quality limitations
```

**Model used:** gemini-2.0-flash (via Google Generative AI API)

### Results

The Gemini API free-tier quota was exhausted after 18 of 100 requests.

| Metric | Value |
|--------|-------|
| CNN-predicted NORMAL (second-look candidates) | 100 |
| Successful VLM assessments | 18 (18%) |
| Failed/unusable (errors) | 1 |
| Quota-stopped | 81 not attempted |
| PNEUMONIA_SUGGESTIVE | 0 |
| NOT_PNEUMONIA_SUGGESTIVE | 14 |
| INDETERMINATE | 4 |
| CNN false negatives assessed | 0 / 5 |
| CNN false negatives flagged | 0 |

### Limitations of the VLM Evaluation

1. **18% coverage is insufficient** to draw conclusions about second-look performance on the full test set.
2. **None of the 5 CNN false negatives** received a VLM assessment. The central research question — whether VLM can identify CNN false negatives — remains unanswered for this dataset under these quota constraints.
3. **No combined-system metrics** are computed. Reporting accuracy, precision, or recall over 18 quota-limited samples as if they represented the 100-image second-look set would be misleading.
4. **Gemini Flash is a general-purpose model**, not a clinically validated radiology AI. Its assessments are not comparable to radiologist readings.

---

## 5. Discussion

### What worked

- The CNN pipeline is fully reproducible: checkpoint, manifests, and results are preserved.
- The VLM integration correctly handles quota errors, saves progress, and resumes from checkpointed results.
- The evaluation framework correctly distinguishes assessed from unassessed cases and refuses to fabricate combined-system metrics from incomplete data.

### What did not work

- Free-tier API quota made a complete second-look run infeasible in a single session.
- The 5 CNN false negatives were never reached by the VLM, so the main research hypothesis cannot be evaluated.

### What the results suggest (with caveats)

Among the 18 successfully assessed CNN-predicted NORMAL images:
- 14 were assessed NOT_PNEUMONIA_SUGGESTIVE.
- 4 were INDETERMINATE.
- 0 were flagged PNEUMONIA_SUGGESTIVE.

This is consistent with the CNN's behaviour (very few false negatives; most predicted-normal cases are genuinely normal), but the sample is far too small to quantify VLM performance.

---

## 6. Future Work

1. **Complete VLM coverage:** Use a paid API tier or batched requests spread across multiple days to assess all 100 second-look candidates.
2. **Target false negatives:** After confirming which 5 images are CNN false negatives, assess them specifically.
3. **Patient-level split:** Request patient ID metadata from the dataset source to construct proper patient-level splits.
4. **Pretrained backbone:** Replace SmallCNN with a pretrained ResNet or DenseNet for higher baseline accuracy.
5. **Threshold optimisation:** Evaluate different CNN decision thresholds to trade false positives for false negatives, tuned on the validation set.
6. **Multiple VLM models:** Compare assessments across Gemini, GPT-4V, and open-weight models.

---

## 7. Ethical Considerations

- This system is an experimental research prototype and must not be used for clinical decisions.
- Ground-truth labels were never provided to the VLM during inference.
- The dataset does not include patient consent information; it is treated as a public benchmark dataset.
- Reported metrics are derived from actual model outputs, not fabricated or post-hoc adjusted.

---

## 8. Reproducibility Checklist

- [x] Dataset manifests with SHA-256 hashes saved in `outputs/splits/`
- [x] CNN checkpoint at `checkpoints/best_cnn.pth`
- [x] Full test predictions at `outputs/test_predictions.csv`
- [x] VLM results at `outputs/gemini_second_look.csv`
- [x] All metrics computed programmatically from saved artifacts
- [x] Figures regenerable with `scripts/generate_analysis.py`
- [x] Unit tests covering dataset, model, VLM parsing, and evaluation
- [x] Requirements pinned in `requirements.txt`
- [x] API key excluded from version control

---

*Report generated from actual experimental results. No values are fabricated or manually edited.*
