"""
SegFormer + frozen Panopticon as parallel decoder input.

Panopticon spatial tokens are projected and fed as a 5th input to
SegFormer's MLP decoder alongside the 4 encoder stages. The decoder
already upsamples all inputs to 1/4 scale and concatenates — Panopticon
features are just another scale.

Input: RGBIR + nDSM concatenated (5-ch early fusion) for SegFormer encoder.
Panopticon receives RGBIR only (4 bands + wavelength IDs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from transformers.models.segformer.modeling_segformer import SegformerMLP


class SegformerDecodeHeadWithPanopticon(nn.Module):
    """SegFormer MLP decode head extended with a 5th Panopticon input.

    Takes 4 encoder stages + 1 Panopticon feature map, projects each to
    decoder_hidden_size, upsamples all to 1/4 scale, concatenates, and
    fuses with a 1x1 conv.
    """

    def __init__(self, config, original_decode_head, panopticon_dim=768):
        super().__init__()
        decoder_hidden_size = config.decoder_hidden_size

        # Keep original 4 stage MLPs and classifier
        self.linear_c = original_decode_head.linear_c
        self.classifier = original_decode_head.classifier

        # Add 5th MLP for Panopticon features
        # SegformerMLP(config, input_dim) -> proj: Linear(input_dim, decoder_hidden_size)
        # Its forward does: flatten(2).transpose(1,2) -> proj -> output
        self.linear_pan = SegformerMLP(config, input_dim=panopticon_dim)

        # Replace fuse conv: 5 * decoder_hidden_size -> decoder_hidden_size
        self.linear_fuse = nn.Conv2d(
            decoder_hidden_size * 5, decoder_hidden_size, kernel_size=1,
            bias=False)
        self.batch_norm = nn.BatchNorm2d(decoder_hidden_size)
        self.relu = nn.ReLU()

    def forward(self, encoder_hidden_states, panopticon_feat):
        """
        Args:
            encoder_hidden_states: tuple of 4 tensors from SegFormer encoder.
            panopticon_feat: (B, 768, Hp, Wp) from frozen Panopticon.
        Returns:
            logits: (B, num_classes, H/4, W/4)
        """
        all_hidden = []

        # Process 4 encoder stages
        # SegformerMLP.forward does flatten+transpose internally,
        # so input is (B, C, H, W) and output is (B, HW, decoder_dim)
        for i, (hidden_state, mlp) in enumerate(
                zip(encoder_hidden_states, self.linear_c)):
            B, C, H, W = hidden_state.shape
            proj = mlp(hidden_state)  # (B, HW, decoder_dim)
            D = proj.shape[-1]
            proj = proj.transpose(1, 2).reshape(B, D, H, W)
            all_hidden.append(proj)

        # Process Panopticon features (5th input)
        B, C, H, W = panopticon_feat.shape
        pan_proj = self.linear_pan(panopticon_feat)  # (B, HW, decoder_dim)
        D = pan_proj.shape[-1]
        pan_proj = pan_proj.transpose(1, 2).reshape(B, D, H, W)
        all_hidden.append(pan_proj)

        # Upsample all to largest spatial size (stage 0 = 1/4 scale)
        target_size = all_hidden[0].shape[-2:]
        for i in range(1, len(all_hidden)):
            all_hidden[i] = F.interpolate(
                all_hidden[i], size=target_size,
                mode='bilinear', align_corners=False)

        # Concatenate (reversed order, matching original SegFormer)
        fused = torch.cat(all_hidden[::-1], dim=1)
        fused = self.linear_fuse(fused)
        fused = self.batch_norm(fused)
        fused = self.relu(fused)

        logits = self.classifier(fused)
        return logits


class SegFormerPanopticon(nn.Module):
    """SegFormer encoder-decoder with frozen Panopticon as parallel decoder input.

    Architecture:
        Input (5-ch) -> SegFormer MiT Encoder (trainable) -> 4 stage features
        Input (4-ch) -> Panopticon (frozen) -> spatial tokens (768, 16, 16)

        4 stages + Panopticon -> MLP Decoder (5 inputs) -> logits
    """

    def __init__(self, num_classes=7, in_channels=5,
                 segformer_pretrained="nvidia/segformer-b0-finetuned-ade-512-512",
                 use_pretrained_weights=True, use_panopticon=True,
                 panopticon_checkpoint=None, panopticon_weights=True):
        super().__init__()
        self.use_panopticon = use_panopticon

        # -- SegFormer encoder-decoder (trainable) ---------------------------
        config = SegformerConfig.from_pretrained(segformer_pretrained)
        config.num_labels = num_classes
        config.num_channels = in_channels

        self.segformer = SegformerForSemanticSegmentation(config)

        # Load pretrained weights where possible
        if segformer_pretrained and use_pretrained_weights:
            pretrained = SegformerForSemanticSegmentation.from_pretrained(
                segformer_pretrained)
            pretrained_sd = pretrained.state_dict()
            model_sd = self.segformer.state_dict()
            for k, v in pretrained_sd.items():
                if k in model_sd and v.shape == model_sd[k].shape:
                    model_sd[k] = v
            self.segformer.load_state_dict(model_sd)

            with torch.no_grad():
                self.segformer.segformer.encoder.patch_embeddings[0].proj.weight[:, :3] = \
                    pretrained.segformer.encoder.patch_embeddings[0].proj.weight
            del pretrained

        self.encoder = self.segformer.segformer.encoder

        if self.use_panopticon:
            # Replace decode head with extended version (5 inputs)
            self.decode_head = SegformerDecodeHeadWithPanopticon(
                config,
                self.segformer.decode_head,
                panopticon_dim=768,
            )
            # Frozen Panopticon
            from core.models.moe_segmentor import PanopticonBackbone
            self.panopticon = PanopticonBackbone(
                weights=panopticon_weights,
                img_size=224,
                checkpoint_path=panopticon_checkpoint,
            )
        else:
            # Use original SegFormer decode head (4 inputs)
            self.decode_head = self.segformer.decode_head

    def forward(self, x, x_ms=None, chn_ids=None, rgb_indices=(0, 1, 2)):
        """
        Args:
            x:       (B, 5, H, W) RGBIR+nDSM for SegFormer encoder.
            x_ms:    (B, 4, H, W) RGBIR for Panopticon.
            chn_ids: (B, C) wavelengths for Panopticon.
        Returns:
            dict with 'logits': (B, num_classes, H, W)
        """
        output_size = x.shape[-2:]

        # SegFormer encoder -> 4 stage features
        encoder_outputs = self.encoder(
            x, output_hidden_states=True, return_dict=True)
        # hidden_states may include initial embedding — take last 4 only
        encoder_hidden_states = encoder_outputs.hidden_states
        if len(encoder_hidden_states) > 4:
            encoder_hidden_states = encoder_hidden_states[-4:]

        # Decode
        if self.use_panopticon and x_ms is not None and chn_ids is not None:
            pan_spatial = self.panopticon(x_ms, chn_ids)
            logits = self.decode_head(encoder_hidden_states, pan_spatial)
        else:
            logits = self.decode_head(encoder_hidden_states)

        # Upsample to input resolution
        logits = F.interpolate(logits, size=output_size,
                               mode='bilinear', align_corners=False)

        return {'logits': logits}
