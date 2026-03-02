"""
Model 5 — Small FC Variant (compact classifier head).

Same convolutional layers as baseline, but the "brain" at the
end (fully connected layers) is much smaller: 128 neurons
instead of 512. Tests if we really need a big classifier.

~1.7M parameters (much smaller than baseline's 3.45M)
"""

import torch.nn as nn
from torch import Tensor


class SmallFCCNN(nn.Module):
    """CNN with a tiny fully-connected head — tests compact classifiers."""

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
        )
        # After 3 conv+pool: (128, 17, 17)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 17 * 17, 128),  # Small head: 128 instead of 512
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pass an image batch through the network."""
        x = self.features(x)
        return self.classifier(x)
