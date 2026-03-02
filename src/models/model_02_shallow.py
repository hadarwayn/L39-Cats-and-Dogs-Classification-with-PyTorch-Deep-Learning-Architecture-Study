"""
Model 2 — Shallow CNN (only 2 convolutional layers).

The simplest possible CNN for this task.
Tests the question: "How little depth can we get away with?"

  Input:  (batch, 3, 150, 150)
  Conv1:  3 -> 32, 3x3 + ReLU + MaxPool  => (batch, 32, 74, 74)
  Conv2:  32 -> 64, 3x3 + ReLU + MaxPool => (batch, 64, 36, 36)
  Flatten + FC(64*36*36 -> 128) + ReLU + FC(128 -> 2)

~10.6M parameters (most in the first FC layer due to large feature map)
"""

import torch.nn as nn
from torch import Tensor


class ShallowCNN(nn.Module):
    """Minimal 2-layer CNN — tests the floor of depth."""

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # After two pool layers: 150/2/2 = 37 (with padding)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 37 * 37, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pass an image batch through the network."""
        x = self.features(x)
        return self.classifier(x)
