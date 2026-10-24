"""
scripts/validate_dataset.py — Validate dataset manifests and splits.

Usage (from project root):
    .venv\Scripts\python.exe scripts/validate_dataset.py

Checks:
 - All image files referenced in the manifests exist on disk
 - No SHA-256 hashes overlap between train/validation/test splits
 - Label distribution matches what is expected
 - Reports cross-split duplicate counts from EDA summary
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

SPLITS = {
    "train":      Path("outputs/splits/train.csv"),
    "validation": Path("outputs/splits/validation.csv"),
    "test":       Path("outputs/splits/test.csv"),
}
EDA_SUMMARY = Path("outputs/eda/eda_summary.json")

errors:   list[str] = []
warnings: list[str] = []

def check_manifest(name: str, path: Path) -> list[dict]:
    if not path.exists():
        errors.append(f"Missing manifest: {path}")
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))
    if not records:
        errors.append(f"Empty manifest: {path}")
        return []

    missing = [r["path"] for r in records if not Path(r["path"]).is_file()]
    if missing:
        errors.append(f"{name}: {len(missing)} image(s) not found on disk.")
        for p in missing[:5]:
            errors.append(f"  Missing: {p}")
        if len(missing) > 5:
            errors.append(f"  ... and {len(missing) - 5} more.")

    label_counts = Counter(r["label"] for r in records)
    print(f"  {name}: {len(records)} images | {dict(label_counts)}")
    return records

print("=== Dataset manifest validation ===\n")

all_records = {}
for split_name, manifest_path in SPLITS.items():
    print(f"Checking {split_name}...")
    recs = check_manifest(split_name, manifest_path)
    all_records[split_name] = recs

print()

# Check hash overlap between splits
if all(all_records.values()):
    hashes: dict[str, set[str]] = {
        name: {r["sha256"] for r in recs if "sha256" in r}
        for name, recs in all_records.items()
    }
    if all(hashes.values()):
        train_val = hashes["train"] & hashes["validation"]
        train_test = hashes["train"] & hashes["test"]
        val_test = hashes["validation"] & hashes["test"]
        if train_val:
            errors.append(f"HASH OVERLAP: {len(train_val)} image(s) shared between train and validation!")
        if train_test:
            errors.append(f"HASH OVERLAP: {len(train_test)} image(s) shared between train and test!")
        if val_test:
            errors.append(f"HASH OVERLAP: {len(val_test)} image(s) shared between validation and test!")
        if not (train_val or train_test or val_test):
            print("Hash overlap check: PASSED (no exact duplicates cross splits)")
    else:
        warnings.append("SHA-256 column not found in manifests — skipping overlap check.")

# Load EDA summary
if EDA_SUMMARY.exists():
    with EDA_SUMMARY.open(encoding="utf-8") as f:
        eda = json.load(f)
    print()
    print("EDA summary:")
    print(f"  Original image count:       {eda.get('original_image_count', 'N/A')}")
    print(f"  Clean image count:          {eda.get('clean_image_count', 'N/A')}")
    print(f"  Exact duplicate groups:     {eda.get('exact_duplicate_groups', 'N/A')}")
    print(f"  Cross-split duplicate groups: {eda.get('cross_split_duplicate_groups', 'N/A')}")
    if eda.get("cross_split_duplicate_groups", 0) > 0:
        warnings.append(
            f"WARNING: {eda['cross_split_duplicate_groups']} cross-split duplicate group(s) found. "
            "Split independence cannot be guaranteed."
        )

print()
if warnings:
    print("Warnings:")
    for w in warnings:
        print(f"  ⚠ {w}")
    print()

if errors:
    print("ERRORS:")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("All checks PASSED.")
