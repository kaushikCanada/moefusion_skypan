"""
MoE Fusion Segmentor

Architecture:
  - Frozen SkySense++ HR (SwinV2-Huge): 4-scale spatial features (RGB input)
  - Frozen Panopticon (ViT-B/14): spatially-resolved spectral features (all bands)
  - Panopticon spatial tokens reshaped to 2D, projected, and fused at Stage 2
  - nDSM Encoder: trainable FCN -> 256-ch at 1/4 scale, concat with Stage 0
  - Lightweight FPN: 4-scale features -> 4 outputs at 256-ch
  - UPerNet Head: multi-scale fusion -> per-pixel logits

Ablation flags (set in config under model:):
  use_panopticon_spatial: true/false  -- Panopticon spatial fusion at Stage 2
  use_ndsm: true/false                -- nDSM height fusion at Stage 0

All components except backbones are pure PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.skysense_backbone import build_skysense_hr_backbone
from core.models.heads import LightweightFPN, UPerNetHead


# -----------------------------------------------------------------------------
# Panopticon wrapper (returns spatial tokens, not GAP)
# -----------------------------------------------------------------------------

class PanopticonBackbone(nn.Module):
    """Frozen Panopticon ViT-B/14 from torchgeo.

    Returns spatial patch tokens reshaped to a 2D feature map.
    For ViT-B/14 @ 224x224: 16x16 = 256 tokens, each 768-dim.
    Output: (B, 768, 16, 16).
    """

    def __init__(self, weights=True, img_size=224, checkpoint_path=None):
        super().__init__()
        from torchgeo.models import panopticon_vitb14, Panopticon_Weights

        if checkpoint_path is not None:
            self.backbone = panopticon_vitb14(weights=None, img_size=img_size)
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            # Fix key prefix: checkpoint has 'blocks.0...' but model
            # expects 'model.blocks.0...'
            model_keys = set(self.backbone.state_dict().keys())
            if any(k.startswith('model.') for k in model_keys) and \
               not any(k.startswith('model.') for k in state_dict.keys()):
                state_dict = {f'model.{k}': v for k, v in state_dict.items()}
            # Remove keys with shape mismatch (e.g. pos_embed at different resolution)
            model_sd = self.backbone.state_dict()
            state_dict = {k: v for k, v in state_dict.items()
                          if k in model_sd and v.shape == model_sd[k].shape}
            missing, unexpected = self.backbone.load_state_dict(
                state_dict, strict=False)
            print(f"[Panopticon] Loaded weights from {checkpoint_path}")
            print(f"  Matched keys: {len(model_keys) - len(missing)}/{len(model_keys)}")
            if missing:
                print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
            if unexpected:
                print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        else:
            w = Panopticon_Weights.VIT_BASE14 if weights else None
            self.backbone = panopticon_vitb14(weights=w, img_size=img_size)

        # Freeze
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.embed_dim = 768
        self.patch_size = 14
        self.img_size = img_size
        self.grid_size = img_size // self.patch_size  # 16 for 224

    def forward(self, x_ms, chn_ids):
        """
        Args:
            x_ms: (B, C, H, W) multispectral image (any spatial size).
            chn_ids: (B, C) wavelengths in nm.
        Returns:
            spatial_feat: (B, 768, grid_size, grid_size) spatial feature map.
        """
        if x_ms.shape[-1] != self.img_size or x_ms.shape[-2] != self.img_size:
            x_ms = F.interpolate(x_ms, size=(self.img_size, self.img_size),
                                 mode='bilinear', align_corners=False)

        x_dict = {"imgs": x_ms, "chn_ids": chn_ids}
        with torch.no_grad():
            tokens = self.backbone.model.forward_features(x_dict)
            # tokens: (B, 1+N, 768) -- CLS + patch tokens
            patch_tokens = tokens[:, 1:, :]  # (B, N, 768)
            B, N, D = patch_tokens.shape
            spatial_feat = patch_tokens.permute(0, 2, 1).reshape(
                B, D, self.grid_size, self.grid_size)
        return spatial_feat


# -----------------------------------------------------------------------------
# Panopticon Spatial Fusion (replaces ScaleGate)
# -----------------------------------------------------------------------------

class PanopticonSpatialFusion(nn.Module):
    """Project Panopticon spatial tokens and fuse with SkySense++ Stage 2.

    Panopticon (768, 16, 16) -> proj -> (target_ch, 16, 16)
    Concat with Stage 2 (1408, 16, 16) -> 1x1 conv -> (1408, 16, 16)
    """

    def __init__(self, panopticon_dim=768, stage2_channels=1408,
                 proj_channels=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(panopticon_dim, proj_channels, 1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(stage2_channels + proj_channels, stage2_channels,
                      1, bias=False),
            nn.BatchNorm2d(stage2_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, stage2_feat, panopticon_spatial):
        """
        Args:
            stage2_feat: (B, 1408, H2, W2) from SkySense++ Stage 2.
            panopticon_spatial: (B, 768, 16, 16) from Panopticon.
        Returns:
            fused: (B, 1408, H2, W2)
        """
        pan_proj = self.proj(panopticon_spatial)
        # Match spatial size to Stage 2
        if pan_proj.shape[-2:] != stage2_feat.shape[-2:]:
            pan_proj = F.interpolate(
                pan_proj, size=stage2_feat.shape[-2:],
                mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([stage2_feat, pan_proj], dim=1))


# -----------------------------------------------------------------------------
# nDSM Encoder
# -----------------------------------------------------------------------------

class NdsmEncoder(nn.Module):
    """Lightweight FCN for nDSM input.

    1-ch input -> despeckle conv -> strided convs -> 256-ch at 1/4 scale.
    """

    def __init__(self, out_channels=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.GELU(),
            nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------------------
# Full MoE Segmentor
# -----------------------------------------------------------------------------

class MoESegmentor(nn.Module):
    """
    Frozen SkySense++ (RGB) + optional frozen Panopticon spatial fusion
    + optional nDSM encoder.

    Ablation flags:
        use_panopticon_spatial: fuse Panopticon spatial tokens at Stage 2
        use_ndsm: fuse nDSM height features at Stage 0
    """

    BACKBONE_CHANNELS = [352, 704, 1408, 2816]

    def __init__(self, num_classes, skysense_weights=None, img_size=224,
                 panopticon_weights=True, panopticon_checkpoint=None,
                 fpn_channels=256, ndsm_channels=256, dropout=0.1,
                 use_panopticon_spatial=True, use_ndsm=True):
        super().__init__()
        self.use_panopticon_spatial = use_panopticon_spatial
        self.use_ndsm = use_ndsm

        # -- Frozen SkySense++ (always on) -----------------------------------
        self.skysense = build_skysense_hr_backbone(
            pretrained_path=skysense_weights,
            img_size=img_size,
            frozen=True,
        )

        # -- Frozen Panopticon (only if spatial fusion enabled) ---------------
        if self.use_panopticon_spatial:
            self.panopticon = PanopticonBackbone(
                weights=panopticon_weights,
                img_size=224,
                checkpoint_path=panopticon_checkpoint,
            )
            self.panopticon_fusion = PanopticonSpatialFusion(
                panopticon_dim=self.panopticon.embed_dim,
                stage2_channels=self.BACKBONE_CHANNELS[2],
                proj_channels=256,
            )

        # -- nDSM encoder (only if enabled) ----------------------------------
        if self.use_ndsm:
            self.ndsm_encoder = NdsmEncoder(out_channels=ndsm_channels)
            self.ndsm_fusion = nn.Sequential(
                nn.Conv2d(self.BACKBONE_CHANNELS[0] + ndsm_channels,
                          self.BACKBONE_CHANNELS[0], 1, bias=False),
                nn.BatchNorm2d(self.BACKBONE_CHANNELS[0]),
                nn.ReLU(inplace=True),
            )

        # -- FPN + Head ------------------------------------------------------
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

    def forward(self, x_msi, chn_ids=None, x_ndsm=None,
                rgb_indices=(0, 1, 2)):
        """
        Args:
            x_msi:       (B, C, H, W)  all multispectral bands.
            chn_ids:     (B, C) wavelengths in nm (needed if panopticon on).
            x_ndsm:      (B, 1, H, W)  nDSM input, or None.
            rgb_indices: tuple of 3 ints -- which bands are R, G, B.
        Returns:
            dict with 'logits': (B, num_classes, H, W)
        """
        output_size = x_msi.shape[-2:]
        x_rgb = x_msi[:, rgb_indices, :, :]

        # 1. Frozen SkySense++ -> 4 multi-scale feature maps
        with torch.no_grad():
            sky_features = list(self.skysense(x_rgb))

        # 2. Panopticon spatial fusion at Stage 2
        if self.use_panopticon_spatial:
            pan_spatial = self.panopticon(x_msi, chn_ids)  # (B, 768, 16, 16)
            sky_features[2] = self.panopticon_fusion(
                sky_features[2], pan_spatial)

        # 3. nDSM fusion at Stage 0
        if self.use_ndsm and x_ndsm is not None:
            ndsm_feat = self.ndsm_encoder(x_ndsm)
            if ndsm_feat.shape[-2:] != sky_features[0].shape[-2:]:
                ndsm_feat = F.interpolate(
                    ndsm_feat, size=sky_features[0].shape[-2:],
                    mode='bilinear', align_corners=False)
            sky_features[0] = self.ndsm_fusion(
                torch.cat([sky_features[0], ndsm_feat], dim=1))

        # 4. FPN -> UPerNet -> logits
        fpn_out = self.fpn(sky_features)
        logits = self.head(fpn_out, output_size=output_size)

        return {'logits': logits}
