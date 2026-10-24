from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import csv
import json
import random
from PIL import Image

ROOT = Path("data/raw/chest_xray")
OUT = Path("outputs")
SEED = 42
CLASSES = {"NORMAL": 0, "PNEUMONIA": 1}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

random.seed(SEED)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def scan_split(split):
    records = []
    for label, class_id in CLASSES.items():
        folder = ROOT / split / label
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue

            with Image.open(path) as im:
                im.verify()

            records.append({
                "path": str(path.resolve()),
                "split": split,
                "label": label,
                "label_id": class_id,
                "sha256": sha256(path),
            })
    return records

all_records = []
for split in ("train", "val", "test"):
    records = scan_split(split)
    all_records.extend(records)
    counts = Counter(r["label"] for r in records)
    print(f"{split}: {len(records)} images | {dict(counts)}")

# Audit exact-content duplicates across the complete dataset.
hash_groups = defaultdict(list)
for record in all_records:
    hash_groups[record["sha256"]].append(record)

duplicate_groups = [
    group for group in hash_groups.values() if len(group) > 1
]
cross_split_groups = [
    group for group in duplicate_groups
    if len({r["split"] for r in group}) > 1
]
conflicting_label_groups = [
    group for group in duplicate_groups
    if len({r["label"] for r in group}) > 1
]

print("\nUnique hashes:", len(hash_groups))
print("Exact duplicate groups:", len(duplicate_groups))
print("Cross-split duplicate groups:", len(cross_split_groups))
print("Duplicate groups with conflicting labels:", len(conflicting_label_groups))

# Remove duplicate copies only within each split. Do not silently remove
# cross-split duplicates; report them so they can be resolved explicitly.
clean_records = []
seen_per_split = set()
for record in all_records:
    key = (record["split"], record["sha256"])
    if key in seen_per_split:
        continue
    seen_per_split.add(key)
    clean_records.append(record)

# Keep the original test split held out. Use train + val as development data.
development = [
    r for r in clean_records if r["split"] in ("train", "val")
]
test_records = [r for r in clean_records if r["split"] == "test"]

# Rebuild an approximately 80/20 stratified train/validation split.
train_records, val_records = [], []
for label in CLASSES:
    items = [r for r in development if r["label"] == label]
    random.shuffle(items)
    n_val = round(0.20 * len(items))
    val_records.extend(items[:n_val])
    train_records.extend(items[n_val:])

# Verify exact hashes don't cross the rebuilt splits.
def hashes(records):
    return {r["sha256"] for r in records}

assert not (hashes(train_records) & hashes(val_records))
assert not (hashes(train_records) & hashes(test_records))
assert not (hashes(val_records) & hashes(test_records))

def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "split", "label", "label_id", "sha256"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

write_csv(OUT / "eda" / "dataset_manifest_original.csv", all_records)
write_csv(OUT / "eda" / "dataset_manifest_clean.csv", clean_records)
write_csv(OUT / "splits" / "train.csv", train_records)
write_csv(OUT / "splits" / "validation.csv", val_records)
write_csv(OUT / "splits" / "test.csv", test_records)

summary = {
    "seed": SEED,
    "original_image_count": len(all_records),
    "clean_image_count": len(clean_records),
    "unique_sha256_count": len(hash_groups),
    "exact_duplicate_groups": len(duplicate_groups),
    "cross_split_duplicate_groups": len(cross_split_groups),
    "conflicting_label_duplicate_groups": len(conflicting_label_groups),
    "train_count": len(train_records),
    "validation_count": len(val_records),
    "test_count": len(test_records),
    "class_mapping": CLASSES,
    "notes": [
        "Stratified by image label, not patient ID.",
        "Exact SHA-256 matching does not detect visually identical images with different encodings.",
        "Cross-split duplicates require manual review before claiming split independence.",
        "Test split is preserved from the original dataset."
    ]
}
(OUT / "eda").mkdir(parents=True, exist_ok=True)
with (OUT / "eda" / "eda_summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nRebuilt splits:")
for name, records in [
    ("train", train_records),
    ("validation", val_records),
    ("test", test_records),
]:
    print(name, len(records), dict(Counter(r["label"] for r in records)))

print("\nSaved manifests and summary under outputs/")
