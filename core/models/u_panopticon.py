"""
U-Panopticon: UNet + frozen Panopticon at bottleneck.

Inspired by U-Prithvi (Kostejn et al., 2025). Trainable UNet encoder-decoder
with frozen Panopticon features injected at the bottleneck via concatenation.
A RandomHalfMaskLayer prevents the decoder from ignoring either branch.

Input: RGBIR + nDSM concatenated (5-ch early fusion).
Panopticon receives RGBIR only (4 bands + wavelength IDs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class RandomHalfMaskLayer(nn.Module):
    """Per-sample random masking of two feature branches during training.

    Adapted from U-Prithvi (Kostejn et al., 2025). For each sample in the
    batch, independently chooses one of three strategies:
      - Zero out branch A (FM features), scale branch B by 2x
      - Zero out branch B (UNet features), scale branch A by 2x
      - Keep both unmodified

    The 2x scaling preserves expected activation magnitude (like dropout).
    During inference: always keeps both (no masking).
    """

    def forward(self, feat_a, feat_b):
        if not self.training:
            return feat_a, feat_b

        B = feat_a.shape[0]
        # Per-sample random strategy
        r = torch.rand(B, 1, 1, 1, device=feat_a.device)

        mask_a = torch.ones_like(feat_a)
        mask_b = torch.ones_like(feat_b)
        scale = torch.ones(B, 1, 1, 1, device=feat_a.device)

        # 1/3: zero out A (FM), scale B by 2
        zero_a = r < (1.0 / 3.0)
        mask_a = torch.where(zero_a, torch.zeros_like(mask_a), mask_a)
        scale_b = torch.where(zero_a, torch.tensor(2.0, device=feat_a.device), scale)

        # 1/3: zero out B (UNet), scale A by 2
        zero_b = (r >= 1.0 / 3.0) & (r < 2.0 / 3.0)
        mask_b = torch.where(zero_b, torch.zeros_like(mask_b), mask_b)
        scale_a = torch.where(zero_b, torch.tensor(2.0, device=feat_a.device), scale)

        # 1/3: keep both (scale=1)
        feat_a = feat_a * mask_a * scale_a
        feat_b = feat_b * mask_b * scale_b
        return feat_a, feat_b


class PanopticonBottleneckFusion(nn.Module):
    """Frozen Panopticon -> proj -> concat with UNet bottleneck -> fuse.

    Panopticon (768, 16, 16) -> proj (proj_ch) -> resize to bottleneck spatial
    Concat with UNet bottleneck -> 1x1 conv back to bottleneck channels.
    """

    def __init__(self, panopticon_dim=768, bottleneck_channels=448,
                 proj_channels=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(panopticon_dim, proj_channels, 1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace=True),
        )
        self.mask_layer = RandomHalfMaskLayer()
        self.fuse = nn.Sequential(
            nn.Conv2d(bottleneck_channels + proj_channels, bottleneck_channels,
                      1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, bottleneck_feat, panopticon_spatial):
        pan_proj = self.proj(panopticon_spatial)
        if pan_proj.shape[-2:] != bottleneck_feat.shape[-2:]:
            pan_proj = F.interpolate(
                pan_proj, size=bottleneck_feat.shape[-2:],
                mode='bilinear', align_corners=False)

        pan_proj, bottleneck_feat = self.mask_layer(pan_proj, bottleneck_feat)
        return self.fuse(torch.cat([bottleneck_feat, pan_proj], dim=1))


class UPanopticon(nn.Module):
    """UNet encoder-decoder with frozen Panopticon injected at bottleneck.

    Architecture:
        Input (5-ch: RGBIR+nDSM) -> UNet Encoder (trainable)
                                          |
                                     bottleneck
                                          |
                                    concat + mask  <-- Panopticon (frozen, RGBIR)
                                          |
                                    UNet Decoder (trainable)
                                          |
                                       logits
    """

    def __init__(self, num_classes=7, in_channels=5,
                 encoder_name="efficientnet-b4",
                 encoder_weights="imagenet",
                 panopticon_checkpoint=None,
                 panopticon_weights=True,
                 proj_channels=256):
        super().__init__()

        # -- Trainable UNet encoder-decoder ----------------------------------
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

        # Get bottleneck channel count from encoder
        # smp encoder stages: the last one is the bottleneck
        encoder_channels = self.unet.encoder.out_channels
        # encoder_channels is like (3, 48, 32, 56, 160, 448) for effb4
        # last element is the bottleneck
        self.bottleneck_channels = encoder_channels[-1]

        # -- Frozen Panopticon -----------------------------------------------
        from core.models.moe_segmentor import PanopticonBackbone
        self.panopticon = PanopticonBackbone(
            weights=panopticon_weights,
            img_size=224,
            checkpoint_path=panopticon_checkpoint,
        )

        # -- Bottleneck fusion -----------------------------------------------
        self.fusion = PanopticonBottleneckFusion(
            panopticon_dim=self.panopticon.embed_dim,
            bottleneck_channels=self.bottleneck_channels,
            proj_channels=proj_channels,
        )

    def forward(self, x, x_ms=None, chn_ids=None, rgb_indices=(0, 1, 2)):
        """
        Args:
            x:           (B, 5, H, W) RGBIR+nDSM concatenated for UNet.
            x_ms:        (B, 4, H, W) RGBIR for Panopticon. If None, extracted
                         from x using rgb_indices + remaining bands.
            chn_ids:     (B, C) wavelengths for Panopticon.
            rgb_indices: which bands in x_ms are RGB (not used here, Panopticon
                         gets all MS bands).
        Returns:
            dict with 'logits': (B, num_classes, H, W)
        """
        # -- UNet encoder ----------------------------------------------------
        features = self.unet.encoder(x)
        # features: list of tensors, last is bottleneck

        # -- Panopticon spatial features -------------------------------------
        if x_ms is not None and chn_ids is not None:
            pan_spatial = self.panopticon(x_ms, chn_ids)

            # Fuse at bottleneck (last encoder feature)
            features[-1] = self.fusion(features[-1], pan_spatial)

        # -- UNet decoder ----------------------------------------------------
        decoder_output = self.unet.decoder(features)
        logits = self.unet.segmentation_head(decoder_output)

        return {'logits': logits}
