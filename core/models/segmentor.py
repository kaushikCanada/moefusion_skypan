"""
Minimal end-to-end segmentation model:
  Frozen SkySense++ HR → Lightweight FPN → UPerNet Head → logits

All pure PyTorch, no mmlab deps (except via skysense_backbone).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.skysense_backbone import build_skysense_hr_backbone
from core.models.heads import LightweightFPN, UPerNetHead


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end segmentation model
# ─────────────────────────────────────────────────────────────────────────────

class SkySenseSegmentor(nn.Module):
    """Frozen SkySense++ HR → FPN → UPerNet → per-pixel logits.

    Args:
        num_classes: number of segmentation classes.
        pretrained_path: path to SkySense++ HR backbone weights.
        img_size: input image size.
        fpn_channels: FPN and UPerNet intermediate channels.
        dropout: dropout in UPerNet head.
    """

    # SwinV2-Huge output channels per stage
    BACKBONE_CHANNELS = [352, 704, 1408, 2816]

    def __init__(self, num_classes, pretrained_path=None, img_size=224,
                 fpn_channels=256, dropout=0.1):
        super().__init__()

        self.backbone = build_skysense_hr_backbone(
            pretrained_path=pretrained_path,
            img_size=img_size,
            frozen=True,
        )

        self.fpn = LightweightFPN(
            in_channels_list=self.BACKBONE_CHANNELS,
            out_channels=fpn_channels,
        )

        self.head = UPerNetHead(
            in_channels=fpn_channels,
            num_levels=4,
            fpn_channels=fpn_channels,
            num_classes=num_classes,
            pool_scales=(1, 2, 3, 6),
            dropout=dropout,
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) RGB image.
        Returns:
            logits: (B, num_classes, H, W) at input resolution.
        """
        output_size = x.shape[-2:]

        with torch.no_grad():
            features = self.backbone(x)

        fpn_out = self.fpn(features)
        logits = self.head(fpn_out, output_size=output_size)

        return logits
