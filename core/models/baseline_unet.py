"""
Baseline: UNet with EfficientNet-B4 encoder (ImageNet pretrained).

Takes RGBIR + nDSM concatenated as 5-channel input.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class BaselineUNet(nn.Module):

    def __init__(self, num_classes=7, in_channels=5,
                 encoder_name="efficientnet-b4",
                 encoder_weights="imagenet"):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

    def forward(self, x):
        """
        Args:
            x: (B, 5, H, W) — RGBIR + nDSM concatenated.
        Returns:
            logits: (B, num_classes, H, W)
        """
        return self.model(x)
