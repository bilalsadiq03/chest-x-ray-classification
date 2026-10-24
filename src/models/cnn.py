"""
CNN model definition for PneumoBackstop.

SmallCNN is a lightweight 4-block convolutional network trained from scratch
on 224×224 RGB chest X-rays. It is not a pretrained medical model and should
not be used for clinical decisions.

Architecture:
  Conv(3→16) + BN + ReLU + MaxPool2d(2)
  Conv(16→32) + BN + ReLU + MaxPool2d(2)
  Conv(32→64) + BN + ReLU + MaxPool2d(2)
  Conv(64→96) + BN + ReLU + AdaptiveAvgPool2d(1,1)
  Dropout(0.3) → Linear(96 → 2)

Output: raw logits; apply softmax for class probabilities.
"""

from pathlib import Path
from typing import Optional

import torch
from torch import nn


class SmallCNN(nn.Module):
    """
    Lightweight CNN for binary chest X-ray classification
    (NORMAL=0 vs PNEUMONIA=1).
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(96, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(
    checkpoint: Optional[str | Path] = None,
    device: str = "cpu",
) -> SmallCNN:
    """
    Instantiate SmallCNN and optionally load a saved checkpoint.

    Args:
        checkpoint: Path to a .pth file saved with ``torch.save(state_dict)``.
                    If None, the model is returned with random weights.
        device:     Torch device string (e.g. ``"cpu"``).

    Returns:
        SmallCNN in eval mode with weights loaded from checkpoint (if given).
    """
    model = SmallCNN()
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(str(path), map_location=device, weights_only=True)
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
