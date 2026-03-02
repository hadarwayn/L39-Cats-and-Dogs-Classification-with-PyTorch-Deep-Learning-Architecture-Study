"""
Model 1 — Baseline CNN (converted from the Keras example).

This is the starting point: a straightforward 4-layer CNN.
Every other model is compared against this baseline.

Architecture walkthrough (for a 150x150 RGB image):
  Input:  (batch, 3, 150, 150)
  Conv1:  3 -> 32 filters, 3x3  => (batch, 32, 148, 148)
  Pool1:  MaxPool 2x2            => (batch, 32, 74, 74)
  Conv2:  32 -> 64, 3x3         => (batch, 64, 72, 72)
  Pool2:  MaxPool 2x2            => (batch, 64, 36, 36)
  Conv3:  64 -> 128, 3x3        => (batch, 128, 34, 34)
  Pool3:  MaxPool 2x2            => (batch, 128, 17, 17)
  Conv4:  128 -> 128, 3x3       => (batch, 128, 15, 15)
  Pool4:  MaxPool 2x2            => (batch, 128, 7, 7)
  Flatten:                       => (batch, 6272)
  FC1:    6272 -> 512 + ReLU     => (batch, 512)
  FC2:    512 -> 2               => (batch, 2)

~3.45M parameters
"""

import torch.nn as nn
from torch import Tensor


class BaselineCNN(nn.Module):
    """Baseline 4-layer CNN — direct conversion from the Keras example."""

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pass an image batch through the network."""
        x = self.features(x)
        return self.classifier(x)
