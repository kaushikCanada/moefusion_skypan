"""
MoE Fusion Segmentor

Architecture:
  - Frozen SkySense++ HR (SwinV2-Huge): 4-scale spatial features (RGB input)
  - Frozen Panopticon (ViT-B/14): global scene vector (all bands input)
  - Scale Gate: Panopticon GAP → MLP → 4 per-scale weights → modulate SKY++ stages
  - nDSM Encoder: trainable FCN → 256-ch at 1/4 scale, concat with gated Stage 0
  - Lightweight FPN: 4-scale features → 4 outputs at 256-ch
  - UPerNet Head: multi-scale fusion → per-pixel logits

All components except backbones are pure PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.skysense_backbone import build_skysense_hr_backbone


# ─────────────────────────────────────────────────────────────────────────────
# Panopticon wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PanopticonBackbone(nn.Module):
    """Frozen Panopticon ViT-B/14 from torchgeo.

    Returns global average pooled feature vector (B, 768).
    """

    def __init__(self, weights=True, img_size=224, checkpoint_path=None):
        super().__init__()
        from torchgeo.models import panopticon_vitb14, Panopticon_Weights

        if checkpoint_path is not None:
            # Load from local .pth file
            self.backbone = panopticon_vitb14(weights=None, img_size=img_size)
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            missing, unexpected = self.backbone.load_state_dict(
                state_dict, strict=False)
            print(f"[Panopticon] Loaded weights from {checkpoint_path}")
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

    def forward(self, x_ms, chn_ids):
        """
        Args:
            x_ms: (B, C, H, W) multispectral image (any spatial size).
            chn_ids: (B, C) wavelengths in nm.
        Returns:
            global_feat: (B, 768) global average pooled features.
        """
        # Resize to 224 if needed — Panopticon ViT-B/14 expects 224×224.
        # Since we GAP over tokens, the exact spatial grid doesn't matter.
        if x_ms.shape[-1] != 224 or x_ms.shape[-2] != 224:
            x_ms = F.interpolate(x_ms, size=(224, 224),
                                 mode='bilinear', align_corners=False)

        x_dict = {"imgs": x_ms, "chn_ids": chn_ids}
        with torch.no_grad():
            tokens = self.backbone.model.forward_features(x_dict)
            assert tokens.ndim == 3 and tokens.shape[-1] == 768, \
                f"Unexpected Panopticon output shape: {tokens.shape}"
            # tokens: (B, 1+N, 768) — CLS + patch tokens
            patch_tokens = tokens[:, 1:, :]  # (B, N, 768)
            global_feat = patch_tokens.mean(dim=1)  # (B, 768)
        return global_feat


# ─────────────────────────────────────────────────────────────────────────────
# Scale Gate
# ─────────────────────────────────────────────────────────────────────────────

class ScaleGate(nn.Module):
    """MLP that produces per-scale weights from Panopticon's global vector.

    Panopticon GAP → Linear → GELU → Linear → Softmax → 4 weights
    """

    def __init__(self, in_dim=768, hidden_dim=128, num_scales=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_scales),
        )

    def forward(self, global_feat):
        """
        Args:
            global_feat: (B, 768)
        Returns:
            weights: (B, 4) softmax-normalized scale weights
        """
        return F.softmax(self.mlp(global_feat), dim=-1) * 4


# ─────────────────────────────────────────────────────────────────────────────
# nDSM Encoder
# ─────────────────────────────────────────────────────────────────────────────

class NdsmEncoder(nn.Module):
    """Lightweight FCN for nDSM input.

    1-ch input → despeckle conv → strided convs → 256-ch at 1/4 scale.
    """

    def __init__(self, out_channels=256):
        super().__init__()
        self.net = nn.Sequential(
            # Despeckle: smooth noise in nDSM
            nn.Conv2d(1, 1, 3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.GELU(),
            # Downsample 1/2
            nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # Downsample 1/4
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # Refine at 1/4
            nn.Conv2d(128, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) nDSM input.
        Returns:
            (B, 256, H/4, W/4)
        """
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight FPN
# ─────────────────────────────────────────────────────────────────────────────

class LightweightFPN(nn.Module):
    """Standard FPN with lateral + top-down + smooth convs."""

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
            features: list of 4 tensors [1/4, 1/8, 1/16, 1/32].
        Returns:
            list of 4 tensors, all out_channels, at original spatial sizes.
        """
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:],
                mode='bilinear', align_corners=False)

        return [conv(lat) for conv, lat in zip(self.smooth_convs, laterals)]


# ─────────────────────────────────────────────────────────────────────────────
# UPerNet Head
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
    """UPerNet decode head."""

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


# ─────────────────────────────────────────────────────────────────────────────
# Full MoE Segmentor
# ─────────────────────────────────────────────────────────────────────────────

class MoESegmentor(nn.Module):
    """
    Frozen SkySense++ (RGB) + Frozen Panopticon (all bands) + nDSM encoder.

    Panopticon provides a global scene vector that gates SkySense++ multi-scale
    features. nDSM features are concatenated at 1/4 scale (Stage 0).
    Gated + nDSM-fused features go through FPN → UPerNet → logits.

    Forward inputs:
        x_msi:       (B, C, H, W)  — all multispectral bands
        chn_ids:     (B, C)         — wavelengths in nm for Panopticon
        x_ndsm:      (B, 1, H, W)  — nDSM (optional, None to skip)
        rgb_indices: tuple of 3 ints — which bands in x_msi are R, G, B
    """

    BACKBONE_CHANNELS = [352, 704, 1408, 2816]

    def __init__(self, num_classes, skysense_weights=None, img_size=224,
                 panopticon_weights=True, panopticon_checkpoint=None,
                 fpn_channels=256, ndsm_channels=256, dropout=0.1):
        super().__init__()

        # ── Frozen backbones ────────────────────────────────────────────
        self.skysense = build_skysense_hr_backbone(
            pretrained_path=skysense_weights,
            img_size=img_size,
            frozen=True,
        )
        self.panopticon = PanopticonBackbone(
            weights=panopticon_weights,
            img_size=224,  # always 224 — input is resized internally
            checkpoint_path=panopticon_checkpoint,
        )

        # ── Scale gate ──────────────────────────────────────────────────
        self.scale_gate = ScaleGate(
            in_dim=self.panopticon.embed_dim,
            hidden_dim=128,
            num_scales=4,
        )

        # ── nDSM encoder ───────────────────────────────────────────────
        self.ndsm_encoder = NdsmEncoder(out_channels=ndsm_channels)

        # ── Fusion projection (Stage 0 + nDSM → Stage 0 channels) ─────
        self.ndsm_fusion = nn.Sequential(
            nn.Conv2d(self.BACKBONE_CHANNELS[0] + ndsm_channels,
                      self.BACKBONE_CHANNELS[0], 1, bias=False),
            nn.BatchNorm2d(self.BACKBONE_CHANNELS[0]),
            nn.ReLU(inplace=True),
        )

        # ── FPN ─────────────────────────────────────────────────────────
        self.fpn = LightweightFPN(
            in_channels_list=self.BACKBONE_CHANNELS,
            out_channels=fpn_channels,
        )

        # ── UPerNet head ────────────────────────────────────────────────
        self.head = UPerNetHead(
            in_channels=fpn_channels,
            num_levels=4,
            fpn_channels=fpn_channels,
            num_classes=num_classes,
            pool_scales=(1, 2, 3, 6),
            dropout=dropout,
        )

    def forward(self, x_msi, chn_ids, x_ndsm=None, rgb_indices=(0, 1, 2)):
        """
        Args:
            x_msi:       (B, C, H, W) all multispectral bands.
            chn_ids:     (B, C) wavelengths in nm for Panopticon.
            x_ndsm:      (B, 1, H, W) nDSM input, or None.
            rgb_indices: tuple of 3 ints — which bands are R, G, B.
        Returns:
            dict with:
                'logits': (B, num_classes, H, W) at input resolution
                'gate_weights': (B, 4) scale gate weights
        """
        output_size = x_msi.shape[-2:]

        # Route bands: RGB for SkySense++, all bands for Panopticon
        x_rgb = x_msi[:, rgb_indices, :, :]  # (B, 3, H, W)

        # 1. Frozen SkySense++ → 4 multi-scale feature maps
        with torch.no_grad():
            sky_features = list(self.skysense(x_rgb))
            # [0]: (B, 352, H/4, W/4)
            # [1]: (B, 704, H/8, W/8)
            # [2]: (B, 1408, H/16, W/16)
            # [3]: (B, 2816, H/32, W/32)

        # 2. Frozen Panopticon → global scene vector (resizes to 224 internally)
        global_feat = self.panopticon(x_msi, chn_ids)  # (B, 768)

        # 3. Scale gate → per-scale weights
        gate_weights = self.scale_gate(global_feat)  # (B, 4)

        # 4. Modulate SkySense++ features with gate weights
        for i in range(4):
            w = gate_weights[:, i].view(-1, 1, 1, 1)  # (B, 1, 1, 1)
            sky_features[i] = sky_features[i] * w

        # 5. nDSM fusion at Stage 0 (1/4 scale)
        if x_ndsm is not None:
            ndsm_feat = self.ndsm_encoder(x_ndsm)  # (B, 256, H/4, W/4)
            # Match spatial size to Stage 0
            if ndsm_feat.shape[-2:] != sky_features[0].shape[-2:]:
                ndsm_feat = F.interpolate(
                    ndsm_feat, size=sky_features[0].shape[-2:],
                    mode='bilinear', align_corners=False)
            sky_features[0] = self.ndsm_fusion(
                torch.cat([sky_features[0], ndsm_feat], dim=1))

        # 6. FPN
        fpn_out = self.fpn(sky_features)

        # 7. UPerNet → logits
        logits = self.head(fpn_out, output_size=output_size)

        return {'logits': logits, 'gate_weights': gate_weights}
