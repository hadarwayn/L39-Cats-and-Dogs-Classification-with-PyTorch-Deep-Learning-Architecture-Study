"""
Model 9 — Transfer Learning with ResNet18.

Instead of learning from scratch, we use a model (ResNet18) that
already learned to see from millions of images (ImageNet).
We freeze all its layers and only train a small custom "head".

It's like hiring an expert photographer and only teaching them
which bin to sort photos into (cat vs dog).

Only ~130K trainable parameters (out of ~11M total).
NOTE: This model REQUIRES RGB input (3 channels).
"""

import torch.nn as nn
from torch import Tensor
from torchvision import models


class TransferResNet18(nn.Module):
    """Pretrained ResNet18 with a frozen backbone and custom head."""

    def __init__(self, input_channels: int = 3) -> None:
        """
        Args:
            input_channels: must be 3 (RGB). Grayscale not supported.
        """
        if input_channels != 3:
            raise ValueError(
                "TransferResNet18 requires RGB input (3 channels). "
                "Grayscale mode is not supported for this model."
            )
        super().__init__()

        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze all backbone layers — don't update their weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace the final classification layer with our custom head
        num_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pass an image batch through the network."""
        return self.backbone(x)
