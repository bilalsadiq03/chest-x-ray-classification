import csv
from pathlib import Path
from collections import Counter

baseline_path = Path("outputs/test_predictions.csv")
vlm_path = Path("outputs/gemini_second_look.csv")
output_path = Path("outputs/second_look_summary.txt")

with baseline_path.open(newline="", encoding="utf-8-sig") as f:
    baseline = {r["path"]: r for r in csv.DictReader(f)}

with vlm_path.open(newline="", encoding="utf-8-sig") as f:
    vlm_rows = list(csv.DictReader(f))

# Only count successfully completed VLM assessments.
successful = [
    r for r in vlm_rows
    if r.get("path") in baseline
    and not r.get("error", "").strip()
    and r.get("assessment", "").strip()
        in {
            "PNEUMONIA_SUGGESTIVE",
            "NOT_PNEUMONIA_SUGGESTIVE",
            "INDETERMINATE",
        }
]

counts = Counter(r["assessment"] for r in successful)
joined = [(r, baseline[r["path"]]) for r in successful]

flagged = [
    (v, b) for v, b in joined
    if v["assessment"] == "PNEUMONIA_SUGGESTIVE"
]
not_flagged = [
    (v, b) for v, b in joined
    if v["assessment"] == "NOT_PNEUMONIA_SUGGESTIVE"
]
indeterminate = [
    (v, b) for v, b in joined
    if v["assessment"] == "INDETERMINATE"
]

# CNN false negatives are images with true PNEUMONIA but CNN prediction NORMAL.
all_cnn_false_negatives = [
    r for r in baseline.values()
    if r["true_label"] == "PNEUMONIA"
    and r["predicted_label"] == "NORMAL"
]

assessed_paths = {r["path"] for r in successful}
assessed_cnn_misses = [
    r for r in all_cnn_false_negatives
    if r["path"] in assessed_paths
]

recovered_misses = [
    r for r in successful
    if r["assessment"] == "PNEUMONIA_SUGGESTIVE"
    and baseline[r["path"]]["true_label"] == "PNEUMONIA"
]

flagged_true_normal = [
    r for r in successful
    if r["assessment"] == "PNEUMONIA_SUGGESTIVE"
    and baseline[r["path"]]["true_label"] == "NORMAL"
]

lines = []
lines.append("PNEUMOBACKSTOP — SECOND-LOOK EXPLORATORY SUMMARY")
lines.append("=" * 52)
lines.append(f"Total held-out test images: {len(baseline)}")
lines.append(f"CNN-predicted NORMAL images: {sum(r['predicted_label'] == 'NORMAL' for r in baseline.values())}")
lines.append(f"Successful VLM assessments: {len(successful)}")
lines.append(f"Rows with errors or unusable assessments: {len(vlm_rows) - len(successful)}")
lines.append("")
lines.append("VLM assessment distribution:")
for label in [
    "PNEUMONIA_SUGGESTIVE",
    "NOT_PNEUMONIA_SUGGESTIVE",
    "INDETERMINATE",
]:
    lines.append(f"  {label}: {counts[label]}")
lines.append("")
lines.append("CNN false-negative review:")
lines.append(f"  Total CNN false negatives in test set: {len(all_cnn_false_negatives)}")
lines.append(f"  CNN false negatives assessed by VLM: {len(assessed_cnn_misses)}")
lines.append(f"  Assessed CNN false negatives flagged by VLM: {len(recovered_misses)}")
lines.append("")
lines.append("Among successfully assessed images:")
lines.append(f"  VLM pneumonia-suggestive: {len(flagged)}")
lines.append(f"    Ground truth PNEUMONIA: {sum(b['true_label'] == 'PNEUMONIA' for v, b in flagged)}")
lines.append(f"    Ground truth NORMAL: {sum(b['true_label'] == 'NORMAL' for v, b in flagged)}")
lines.append(f"  VLM not-pneumonia-suggestive: {len(not_flagged)}")
lines.append(f"  VLM indeterminate: {len(indeterminate)}")
lines.append("")
lines.append("Flagged case details:")
for r in recovered_misses:
    lines.append(f"  RECOVERED CNN MISS: {r['path']}")
for r in flagged_true_normal:
    lines.append(f"  FLAGGED BUT GROUND TRUTH NORMAL: {r['path']}")

lines.append("")
lines.append(
    "Interpretation: this is a small, quota-limited exploratory subset. "
    "Do not generalize these results to the full test set or clinical use."
)

report = "\n".join(lines)
output_path.write_text(report, encoding="utf-8")
print(report)
print(f"\nSaved summary: {output_path}")
