"""
Unit tests for CNN model architecture and inference.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.cnn import SmallCNN, build_model


class TestSmallCNN:

    def test_output_shape(self):
        model = SmallCNN()
        model.eval()
        x = torch.rand(4, 3, 224, 224)  # batch of 4, RGB, 224x224
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 2), f"Expected (4, 2), got {out.shape}"

    def test_single_image(self):
        model = SmallCNN()
        model.eval()
        x = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)

    def test_softmax_probabilities_sum_to_one(self):
        model = SmallCNN()
        model.eval()
        x = torch.rand(8, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_probabilities_are_in_zero_one(self):
        model = SmallCNN()
        model.eval()
        x = torch.rand(4, 3, 224, 224)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
        assert float(probs.min()) >= 0.0
        assert float(probs.max()) <= 1.0

    def test_build_model_without_checkpoint(self):
        model = build_model(checkpoint=None)
        assert isinstance(model, SmallCNN)

    def test_build_model_missing_checkpoint_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_model(checkpoint=tmp_path / "nonexistent.pth")

    def test_build_model_with_checkpoint(self, tmp_path):
        """Save and reload a checkpoint; verify weights are identical."""
        model_a = SmallCNN()
        ckpt_path = tmp_path / "test.pth"
        torch.save(model_a.state_dict(), str(ckpt_path))

        model_b = build_model(checkpoint=ckpt_path)
        for name, param in model_a.named_parameters():
            assert torch.allclose(param, dict(model_b.named_parameters())[name])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
