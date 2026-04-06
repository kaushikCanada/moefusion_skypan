"""
Panopticon Linear Probe.

Frozen Panopticon encoder + single 1x1 conv classifier.
Shows what Panopticon's spectral features can do alone,
without any complex decoder.

Input: RGBIR multispectral (4 bands + wavelength IDs).
Output: per-pixel logits upsampled to input resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PanopticonLinearProbe(nn.Module):
    """Frozen Panopticon -> 1x1 conv classifier.

    Panopticon produces (B, 768, 16, 16) spatial tokens.
    A single 1x1 conv maps 768 -> num_classes.
    Output is upsampled to input resolution.
    """

    def __init__(self, num_classes=7, panopticon_checkpoint=None,
                 panopticon_weights=True):
        super().__init__()

        from core.models.moe_segmentor import PanopticonBackbone
        self.panopticon = PanopticonBackbone(
            weights=panopticon_weights,
            img_size=224,
            checkpoint_path=panopticon_checkpoint,
        )

        self.classifier = nn.Conv2d(self.panopticon.embed_dim, num_classes,
                                     kernel_size=1)

    def forward(self, x_ms, chn_ids):
        """
        Args:
            x_ms:    (B, C, H, W) multispectral bands.
            chn_ids: (B, C) wavelengths in nm.
        Returns:
            dict with 'logits': (B, num_classes, H, W)
        """
        output_size = x_ms.shape[-2:]

        pan_spatial = self.panopticon(x_ms, chn_ids)  # (B, 768, 16, 16)
        logits = self.classifier(pan_spatial)  # (B, num_classes, 16, 16)

        logits = F.interpolate(logits, size=output_size,
                               mode='bilinear', align_corners=False)

        return {'logits': logits}
