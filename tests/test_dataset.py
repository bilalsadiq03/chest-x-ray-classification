"""
Unit tests for preprocessing, dataset loading, and manifest validation.
"""

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import ChestXrayDataset, CLASS_TO_ID, IMAGE_SIZE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_tiny_manifest(tmp_path: Path, n_normal: int = 2, n_pneumonia: int = 3) -> Path:
    """
    Create a valid CSV manifest pointing to tiny synthetic images.
    Returns the path to the manifest CSV.
    """
    from PIL import Image

    manifest_path = tmp_path / "manifest.csv"
    rows = []

    for cls, label in [("NORMAL", "NORMAL"), ("PNEUMONIA", "PNEUMONIA")]:
        n = n_normal if cls == "NORMAL" else n_pneumonia
        folder = tmp_path / cls
        folder.mkdir(exist_ok=True)
        for i in range(n):
            img_path = folder / f"{cls}_{i}.jpg"
            # Create a minimal valid JPEG (random pixels)
            arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(img_path)
            rows.append({"path": str(img_path), "label": label, "label_id": CLASS_TO_ID[label]})

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "label_id"])
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestChestXrayDataset:

    def test_loads_correct_number_of_samples(self, tmp_path):
        manifest = make_tiny_manifest(tmp_path, n_normal=2, n_pneumonia=3)
        ds = ChestXrayDataset(str(manifest), training=False)
        assert len(ds) == 5

    def test_label_ids_are_correct(self, tmp_path):
        manifest = make_tiny_manifest(tmp_path, n_normal=2, n_pneumonia=2)
        ds = ChestXrayDataset(str(manifest), training=False)
        labels = [ds[i][1].item() for i in range(len(ds))]
        assert set(labels) == {0, 1}  # NORMAL=0, PNEUMONIA=1

    def test_output_tensor_shape(self, tmp_path):
        manifest = make_tiny_manifest(tmp_path, n_normal=1, n_pneumonia=1)
        ds = ChestXrayDataset(str(manifest), training=False)
        tensor, label = ds[0]
        assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert label.ndim == 0

    def test_image_normalization_applied(self, tmp_path):
        manifest = make_tiny_manifest(tmp_path, n_normal=2, n_pneumonia=0)
        ds = ChestXrayDataset(str(manifest), training=False)
        tensor, _ = ds[0]
        # After ImageNet normalization some values should be < 0 or > 1
        assert float(tensor.min()) < 0 or float(tensor.max()) > 1

    def test_unknown_label_raises(self, tmp_path):
        from PIL import Image
        img_path = tmp_path / "bad.jpg"
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(img_path)
        manifest_path = tmp_path / "bad_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "label_id"])
            writer.writeheader()
            writer.writerow({"path": str(img_path), "label": "UNKNOWN", "label_id": 99})
        with pytest.raises(ValueError, match="Unknown label"):
            ChestXrayDataset(str(manifest_path), training=False)

    def test_missing_file_raises(self, tmp_path):
        manifest_path = tmp_path / "missing.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "label_id"])
            writer.writeheader()
            writer.writerow({
                "path": str(tmp_path / "nonexistent.jpg"),
                "label": "NORMAL",
                "label_id": 0,
            })
        with pytest.raises(FileNotFoundError):
            ChestXrayDataset(str(manifest_path), training=False)

    def test_empty_manifest_raises(self, tmp_path):
        manifest_path = tmp_path / "empty.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "label_id"])
            writer.writeheader()
        with pytest.raises(ValueError, match="Empty manifest"):
            ChestXrayDataset(str(manifest_path), training=False)


# ---------------------------------------------------------------------------
# Class mapping tests
# ---------------------------------------------------------------------------

class TestClassMapping:

    def test_normal_maps_to_zero(self):
        assert CLASS_TO_ID["NORMAL"] == 0

    def test_pneumonia_maps_to_one(self):
        assert CLASS_TO_ID["PNEUMONIA"] == 1

    def test_image_size_is_224(self):
        assert IMAGE_SIZE == 224


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
