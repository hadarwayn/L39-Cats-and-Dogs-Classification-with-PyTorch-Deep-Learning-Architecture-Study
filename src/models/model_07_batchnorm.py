"""
Model 7 — CNN + BatchNorm (normalisation technique).

BatchNorm is like a "reset button" between layers. After each
convolution, it adjusts all the numbers so they stay in a nice
range. This helps the model train faster and more stably.

Order per block: Conv -> BatchNorm -> ReLU -> MaxPool

~3.46M parameters (slightly more than baseline due to BN params)
"""

import torch.nn as nn
from torch import Tensor


class BatchNormCNN(nn.Module):
    """Baseline CNN + BatchNorm after every conv — tests training stability."""

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
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
