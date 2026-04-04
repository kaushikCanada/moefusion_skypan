"""Shared FPN and segmentation head modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightFPN(nn.Module):
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
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:],
                mode='bilinear', align_corners=False)
        return [conv(lat) for conv, lat in zip(self.smooth_convs, laterals)]


class PPM(nn.Module):
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
    def __init__(self, in_channels=256, num_levels=4, fpn_channels=256,
                 num_classes=6, pool_scales=(1, 2, 3, 6), dropout=0.1):
        super().__init__()
        self.ppm = PPM(in_channels, in_channels // 4, pool_scales)
        ppm_out_ch = in_channels + in_channels // 4 * len(pool_scales)
        self.ppm_bottleneck = nn.Sequential(
            nn.Conv2d(ppm_out_ch, fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )
        self.fpn_bottlenecks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, fpn_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_levels - 1)
        ])
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fpn_channels * num_levels, fpn_channels, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(fpn_channels, num_classes, 1)

    def forward(self, fpn_features, output_size=None):
        target_size = fpn_features[0].shape[-2:]
        ppm_out = self.ppm(fpn_features[-1])
        ppm_out = self.ppm_bottleneck(ppm_out)
        fpn_outs = []
        for i in range(len(fpn_features) - 1):
            out = self.fpn_bottlenecks[i](fpn_features[i])
            out = F.interpolate(out, size=target_size,
                                mode='bilinear', align_corners=False)
            fpn_outs.append(out)
        ppm_out = F.interpolate(ppm_out, size=target_size,
                                mode='bilinear', align_corners=False)
        fpn_outs.append(ppm_out)
        fused = torch.cat(fpn_outs, dim=1)
        fused = self.fusion_conv(fused)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        if output_size is not None:
            logits = F.interpolate(logits, size=output_size,
                                   mode='bilinear', align_corners=False)
        return logits
