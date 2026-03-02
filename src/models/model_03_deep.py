"""
Model 3 — Deep CNN (6 convolutional layers).

Goes deeper than the baseline to test if more layers
help the model learn more complex patterns.

  Conv layers: 32 -> 64 -> 128 -> 128 -> 256 -> 256
  MaxPool after layers 2, 4, and 6
  Keeps spatial size manageable with selective pooling.

~4-5M parameters
"""

import torch.nn as nn
from torch import Tensor


class DeepCNN(nn.Module):
    """6-layer deep CNN — tests the impact of depth on learning."""

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 150 -> 148 -> 74
            nn.Conv2d(input_channels, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 72 -> 70 -> 35
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: 33 -> 31 -> 15
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # After all pooling: (256, 15, 15)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 15 * 15, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pass an image batch through the network."""
        x = self.features(x)
        return self.classifier(x)
