"""
Model 10 — Lightweight Efficient CNN.

The smallest model in our study. Uses Global Average Pooling
instead of giant fully-connected layers, which dramatically
reduces the parameter count.

Global Average Pooling works like this: instead of flattening
a 64x4x4 feature map into 1024 numbers, it averages each
channel down to a single number — giving just 64 numbers.

Only ~25-30K parameters (100x fewer than the baseline!)
"""

import torch.nn as nn
from torch import Tensor


class LightweightCNN(nn.Module):
    """Tiny CNN with Global Average Pooling — fewest parameters."""

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Global Average Pooling: reduces any spatial size to 1x1
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Tiny classifier: just 64 -> 2
        self.classifier = nn.Linear(64, 2)

    def forward(self, x: Tensor) -> Tensor:
        """Pass an image batch through the network."""
        x = self.features(x)
        x = self.gap(x)            # (batch, 64, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 64)
        return self.classifier(x)
