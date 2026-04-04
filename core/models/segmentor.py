"""
Minimal end-to-end segmentation model:
  Frozen SkySense++ HR → Lightweight FPN → UPerNet Head → logits

All pure PyTorch, no mmlab deps (except via skysense_backbone).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.skysense_backbone import build_skysense_hr_backbone


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight FPN
# ─────────────────────────────────────────────────────────────────────────────

class LightweightFPN(nn.Module):
    """Takes multi-scale features from backbone and produces 4 feature maps
    all with `out_channels` channels at scales 1/4, 1/8, 1/16, 1/32 of input.

    Backbone outputs (for SwinV2-Huge @ 224):
        Stage 0: (B, 352,  56, 56)  = 1/4
        Stage 1: (B, 704,  28, 28)  = 1/8
        Stage 2: (B, 1408, 14, 14)  = 1/16
        Stage 3: (B, 2816,  7,  7)  = 1/32
    """

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1)
            for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features):
        """
        Args:
            features: list of 4 tensors from backbone stages.
        Returns:
            list of 4 tensors, all with out_channels, at original spatial sizes.
        """
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:],
                mode='bilinear', align_corners=False)

        outs = [conv(lat) for conv, lat in zip(self.smooth_convs, laterals)]
        return outs


# ─────────────────────────────────────────────────────────────────────────────
# UPerNet Head (pure PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

class PPM(nn.Module):
    """Pyramid Pooling Module."""

    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for s in pool_scales
        ])

    def forward(self, x):
        h, w = x.shape[-2:]
        outs = [x]
        for stage in self.stages:
            pooled = stage(x)
            outs.append(F.interpolate(
                pooled, size=(h, w), mode='bilinear', align_corners=False))
        return torch.cat(outs, dim=1)


class UPerNetHead(nn.Module):
    """UPerNet decode head.

    Args:
        in_channels: channels of each FPN level (all same after FPN).
        num_levels: number of FPN levels (4).
        fpn_channels: intermediate channel dim.
        num_classes: output classes.
        pool_scales: PPM pool scales.
        dropout: dropout ratio before final classifier.
    """

    def __init__(self, in_channels=256, num_levels=4, fpn_channels=256,
                 num_classes=6, pool_scales=(1, 2, 3, 6), dropout=0.1):
        super().__init__()

        # PPM on deepest level
        self.ppm = PPM(in_channels, in_channels // 4, pool_scales)
        ppm_out_ch = in_channels + in_channels // 4 * len(pool_scales)
        self.ppm_bottleneck = nn.Sequential(
            nn.Conv2d(ppm_out_ch, fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )

        # Per-level bottleneck (for levels 0..n-2, level n-1 uses PPM)
        self.fpn_bottlenecks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, fpn_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_levels - 1)
        ])

        # Final fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fpn_channels * num_levels, fpn_channels, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(fpn_channels, num_classes, 1)

    def forward(self, fpn_features, output_size=None):
        """
        Args:
            fpn_features: list of 4 FPN outputs [1/4, 1/8, 1/16, 1/32].
            output_size: (H, W) to upsample logits to. If None, returns at
                         1/4 scale.
        Returns:
            logits: (B, num_classes, H, W)
        """
        target_size = fpn_features[0].shape[-2:]

        # PPM on deepest level
        ppm_out = self.ppm(fpn_features[-1])
        ppm_out = self.ppm_bottleneck(ppm_out)

        # Process each level and upsample to 1/4 scale
        fpn_outs = []
        for i in range(len(fpn_features) - 1):
            out = self.fpn_bottlenecks[i](fpn_features[i])
            out = F.interpolate(out, size=target_size,
                                mode='bilinear', align_corners=False)
            fpn_outs.append(out)

        ppm_out = F.interpolate(ppm_out, size=target_size,
                                mode='bilinear', align_corners=False)
        fpn_outs.append(ppm_out)

        # Fuse all levels
        fused = torch.cat(fpn_outs, dim=1)
        fused = self.fusion_conv(fused)
        fused = self.dropout(fused)
        logits = self.classifier(fused)

        if output_size is not None:
            logits = F.interpolate(logits, size=output_size,
                                   mode='bilinear', align_corners=False)

        return logits


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
