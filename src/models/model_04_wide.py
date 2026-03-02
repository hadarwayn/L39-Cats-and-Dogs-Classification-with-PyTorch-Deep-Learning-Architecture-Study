"""
Model 4 — Wide CNN (fewer layers, more filters per layer).

Instead of going deep, this model goes wide: each layer has
more filters to capture more patterns at each level.
Tests: "Is it better to be wide or deep?"

  Conv1:  3 -> 64, 3x3 + ReLU + MaxPool   => (batch, 64, 74, 74)
  Conv2:  64 -> 128, 3x3 + ReLU + MaxPool  => (batch, 128, 36, 36)
  Conv3:  128 -> 256, 3x3 + ReLU + MaxPool => (batch, 256, 17, 17)
  Flatten + FC(256*17*17 -> 512) + ReLU + FC(512 -> 2)

~38M parameters (wider layers = more weights)
"""

import torch.nn as nn
from torch import Tensor


class WideCNN(nn.Module):
    """3-layer wide CNN — more filters per layer than baseline."""

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # After 3 conv+pool: (256, 17, 17)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 17 * 17, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pass an image batch through the network."""
        x = self.features(x)
        return self.classifier(x)
