"""
Model 8 — LeNet-Style CNN (classic 1998 architecture).

Inspired by Yann LeCun's LeNet-5, one of the first CNNs ever.
Key differences from modern models:
  - 5x5 kernels (bigger "eyes" looking at the image)
  - Average Pooling (smoother downsampling)
  - Classic FC sizes: 120 -> 84 -> output

~200K parameters (very small by modern standards)
"""

import torch.nn as nn
from torch import Tensor


class LeNetCNN(nn.Module):
    """LeNet-inspired CNN with 5x5 kernels and average pooling."""

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Layer 1: big 5x5 filters
            nn.Conv2d(input_channels, 6, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(2),
            # Layer 2: more 5x5 filters
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        # After 2 conv+pool with 5x5: (16, 34, 34)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 34 * 34, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pass an image batch through the network."""
        x = self.features(x)
        return self.classifier(x)
