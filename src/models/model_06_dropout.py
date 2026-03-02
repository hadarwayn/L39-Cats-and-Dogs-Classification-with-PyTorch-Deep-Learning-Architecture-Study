"""
Model 6 — CNN + Dropout (regularisation technique).

Same architecture as the baseline, but with Dropout layers added.
Dropout randomly "turns off" some neurons during training,
like studying with a blindfold sometimes — it forces the model
to not rely too much on any single neuron.

Dropout(0.5) = 50% of neurons turned off after Flatten
Dropout(0.3) = 30% turned off after the first FC layer

~3.45M parameters (same as baseline — dropout adds no new weights)
"""

import torch.nn as nn
from torch import Tensor


class DropoutCNN(nn.Module):
    """Baseline CNN + Dropout — tests regularisation via dropout."""

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
            nn.Dropout(0.5),  # Turn off 50% of neurons
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Turn off 30% of neurons
            nn.Linear(512, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pass an image batch through the network."""
        x = self.features(x)
        return self.classifier(x)
