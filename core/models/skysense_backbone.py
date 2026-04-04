"""
SkySense++ HR backbone (SwinTransformerV2-Huge with MSL extensions).

Adapted from: https://github.com/kang-wu/SkySensePlusPlus
Imports patched for OpenMMLab v2.x (mmcv 2.x + mmengine + mmpretrain).

Original: mmcv v1.x / mmcls v0.x
Patched:  mmcv v2.x / mmengine / mmpretrain v1.x
"""

from copy import deepcopy
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

# ── mmcv v2.x / mmengine imports (patched from v1.x) ────────────────────────
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.runner import load_state_dict, CheckpointLoader
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.init import trunc_normal_

# ── mmpretrain (successor to mmcls) ─────────────────────────────────────────
from mmpretrain.models.utils import (
    PatchMerging, ShiftWindowMSA, WindowMSAV2,
    resize_pos_embed, to_2tuple,
)
from mmpretrain.models.backbones.base_backbone import BaseBackbone
from mmcv.cnn.bricks.transformer import MultiheadAttention


# ─────────────────────────────────────────────────────────────────────────────
# SwinBlockV2
# ─────────────────────────────────────────────────────────────────────────────

class SwinBlockV2(BaseModule):
    """Swin Transformer V2 block with post-normalization."""

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size=8,
                 shift=False,
                 extra_norm=False,
                 ffn_ratio=4.,
                 drop_path=0.,
                 pad_small_map=False,
                 attn_cfgs=dict(),
                 ffn_cfgs=dict(),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained_window_size=0,
                 init_cfg=None):

        super(SwinBlockV2, self).__init__(init_cfg)
        self.with_cp = with_cp
        self.extra_norm = extra_norm

        _attn_cfgs = {
            'embed_dims': embed_dims,
            'num_heads': num_heads,
            'shift_size': window_size // 2 if shift else 0,
            'window_size': window_size,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'pad_small_map': pad_small_map,
            'window_msa': WindowMSAV2,
            'pretrained_window_size': to_2tuple(pretrained_window_size),
            **attn_cfgs
        }
        self.attn = ShiftWindowMSA(**_attn_cfgs)
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        _ffn_cfgs = {
            'embed_dims': embed_dims,
            'feedforward_channels': int(embed_dims * ffn_ratio),
            'num_fcs': 2,
            'ffn_drop': 0,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'act_cfg': dict(type='GELU'),
            'add_identity': False,
            **ffn_cfgs
        }
        self.ffn = FFN(**_ffn_cfgs)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        if self.extra_norm:
            self.norm3 = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.attn(x, hw_shape)
            x = self.norm1(x)
            x = x + identity

            identity = x
            x = self.ffn(x)
            x = self.norm2(x)
            x = x + identity

            if self.extra_norm:
                x = self.norm3(x)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


# ─────────────────────────────────────────────────────────────────────────────
# SwinBlockV2Sequence
# ─────────────────────────────────────────────────────────────────────────────

class SwinBlockV2Sequence(BaseModule):
    """Sequence of Swin V2 blocks with optional downsample."""

    def __init__(self,
                 embed_dims,
                 depth,
                 num_heads,
                 window_size=8,
                 downsample=False,
                 downsample_cfg=dict(),
                 drop_paths=0.,
                 block_cfgs=dict(),
                 with_cp=False,
                 pad_small_map=False,
                 extra_norm_every_n_blocks=0,
                 pretrained_window_size=0,
                 init_cfg=None):
        super().__init__(init_cfg)

        if not isinstance(drop_paths, Sequence):
            drop_paths = [drop_paths] * depth

        if not isinstance(block_cfgs, Sequence):
            block_cfgs = [deepcopy(block_cfgs) for _ in range(depth)]

        if downsample:
            self.out_channels = 2 * embed_dims
            _downsample_cfg = {
                'in_channels': embed_dims,
                'out_channels': self.out_channels,
                'norm_cfg': dict(type='LN'),
                **downsample_cfg
            }
            self.downsample = PatchMerging(**_downsample_cfg)
        else:
            self.out_channels = embed_dims
            self.downsample = None

        self.blocks = ModuleList()
        for i in range(depth):
            extra_norm = True if extra_norm_every_n_blocks and \
                (i + 1) % extra_norm_every_n_blocks == 0 else False
            _block_cfg = {
                'embed_dims': self.out_channels,
                'num_heads': num_heads,
                'window_size': window_size,
                'shift': False if i % 2 == 0 else True,
                'extra_norm': extra_norm,
                'drop_path': drop_paths[i],
                'with_cp': with_cp,
                'pad_small_map': pad_small_map,
                'pretrained_window_size': pretrained_window_size,
                **block_cfgs[i]
            }
            block = SwinBlockV2(**_block_cfg)
            self.blocks.append(block)

    def forward(self, x, in_shape):
        if self.downsample:
            x, out_shape = self.downsample(x, in_shape)
        else:
            out_shape = in_shape

        for block in self.blocks:
            x = block(x, out_shape)

        return x, out_shape


# ─────────────────────────────────────────────────────────────────────────────
# SwinTransformerV2
# ─────────────────────────────────────────────────────────────────────────────

class SwinTransformerV2(BaseBackbone):
    """Swin Transformer V2 backbone."""

    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': 96,
                         'depths':     [2, 2,  6,  2],
                         'num_heads':  [3, 6, 12, 24],
                         'extra_norm_every_n_blocks': 0}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': 96,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [3, 6, 12, 24],
                         'extra_norm_every_n_blocks': 0}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': 128,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [4, 8, 16, 32],
                         'extra_norm_every_n_blocks': 0}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': 192,
                         'depths':     [2,  2, 18,  2],
                         'num_heads':  [6, 12, 24, 48],
                         'extra_norm_every_n_blocks': 0}),
        **dict.fromkeys(['h', 'huge'],
                        {'embed_dims': 352,
                         'depths':     [2,  2, 18,  2],
                         'num_heads':  [8, 16, 32, 64],
                         'extra_norm_every_n_blocks': 6}),
        **dict.fromkeys(['g', 'giant'],
                        {'embed_dims': 512,
                         'depths':     [2,  2, 42,  4],
                         'num_heads':  [16, 32, 64, 128],
                         'extra_norm_every_n_blocks': 6}),
    }

    _version = 1
    num_extra_tokens = 0

    def __init__(self,
                 arch='tiny',
                 img_size=256,
                 patch_size=4,
                 in_channels=3,
                 vocabulary_size=128,
                 window_size=8,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 out_indices=(3, ),
                 use_abs_pos_embed=False,
                 interpolate_mode='bicubic',
                 with_cp=False,
                 frozen_stages=-1,
                 norm_eval=False,
                 pad_small_map=False,
                 norm_cfg=dict(type='LN'),
                 stage_cfgs=dict(downsample_cfg=dict(use_post_norm=True)),
                 patch_cfg=dict(),
                 pretrained_window_sizes=[0, 0, 0, 0],
                 init_cfg=None):
        super(SwinTransformerV2, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'depths', 'num_heads',
                'extra_norm_every_n_blocks'
            }
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.vocabulary_size = vocabulary_size + 1
        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']
        self.extra_norm_every_n_blocks = self.arch_settings[
            'extra_norm_every_n_blocks']
        self.num_layers = len(self.depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        self.interpolate_mode = interpolate_mode
        self.frozen_stages = frozen_stages

        if isinstance(window_size, int):
            self.window_sizes = [window_size for _ in range(self.num_layers)]
        elif isinstance(window_size, Sequence):
            assert len(window_size) == self.num_layers
            self.window_sizes = window_size
        else:
            raise TypeError('window_size should be a Sequence or int.')

        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            norm_cfg=dict(type='LN'),
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        self.patch_size = patch_size

        if self.use_abs_pos_embed:
            num_patches = self.patch_resolution[0] * self.patch_resolution[1]
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.embed_dims))
            self._register_load_state_dict_pre_hook(
                self._prepare_abs_pos_embed)

        self._register_load_state_dict_pre_hook(self._delete_reinit_params)

        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.norm_eval = norm_eval

        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.stages = ModuleList()
        embed_dims = [self.embed_dims]
        for i, (depth, num_heads) in enumerate(zip(self.depths,
                                                   self.num_heads)):
            if isinstance(stage_cfgs, Sequence):
                stage_cfg = stage_cfgs[i]
            else:
                stage_cfg = deepcopy(stage_cfgs)
            downsample = True if i > 0 else False
            _stage_cfg = {
                'embed_dims': embed_dims[-1],
                'depth': depth,
                'num_heads': num_heads,
                'window_size': self.window_sizes[i],
                'downsample': downsample,
                'drop_paths': dpr[:depth],
                'with_cp': with_cp,
                'pad_small_map': pad_small_map,
                'extra_norm_every_n_blocks': self.extra_norm_every_n_blocks,
                'pretrained_window_size': pretrained_window_sizes[i],
                **stage_cfg
            }

            stage = SwinBlockV2Sequence(**_stage_cfg)
            self.stages.append(stage)

            dpr = dpr[depth:]
            embed_dims.append(stage.out_channels)

        for i in out_indices:
            if norm_cfg is not None:
                norm_layer = build_norm_layer(norm_cfg, embed_dims[i + 1])[1]
            else:
                norm_layer = nn.Identity()
            self.add_module(f'norm{i}', norm_layer)

    def init_weights(self):
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            import logging
            logger = logging.getLogger(__name__)
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            load_state_dict(self, state_dict, strict=False, logger=logger)
            return
        else:
            super(SwinTransformerV2, self).init_weights()
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + resize_pos_embed(
                self.absolute_pos_embed, self.patch_resolution, hw_shape,
                self.interpolate_mode, self.num_extra_tokens)
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape,
                               stage.out_channels).permute(0, 3, 1,
                                                           2).contiguous()
                outs.append(out)

        return outs

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(0, self.frozen_stages + 1):
            m = self.stages[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        for i in self.out_indices:
            if i <= self.frozen_stages:
                for param in getattr(self, f'norm{i}').parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(SwinTransformerV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _prepare_abs_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'absolute_pos_embed'
        if name not in state_dict.keys():
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        if self.absolute_pos_embed.shape != ckpt_pos_embed_shape:
            ckpt_pos_embed_shape = to_2tuple(
                int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
            pos_embed_shape = self.patch_embed.init_out_size

            state_dict[name] = resize_pos_embed(state_dict[name],
                                                ckpt_pos_embed_shape,
                                                pos_embed_shape,
                                                self.interpolate_mode,
                                                self.num_extra_tokens)

    def _delete_reinit_params(self, state_dict, prefix, *args, **kwargs):
        relative_position_index_keys = [
            k for k in state_dict.keys() if 'relative_position_index' in k
        ]
        for k in relative_position_index_keys:
            del state_dict[k]

        relative_position_index_keys = [
            k for k in state_dict.keys() if 'relative_coords_table' in k
        ]
        for k in relative_position_index_keys:
            del state_dict[k]


# ─────────────────────────────────────────────────────────────────────────────
# Proj_MHSA
# ─────────────────────────────────────────────────────────────────────────────

class Proj_MHSA(nn.Module):

    def __init__(self, embed_dims, proj_dims, num_heads=16,
                 batch_first=True, bias=True):
        super().__init__()
        self.proj_in = nn.Linear(in_features=embed_dims, out_features=proj_dims)
        self.attn = MultiheadAttention(
            embed_dims=proj_dims,
            num_heads=num_heads,
            batch_first=batch_first,
            bias=bias
        )
        self.proj_out = nn.Linear(in_features=proj_dims, out_features=embed_dims)

    def forward(self, x):
        x = self.proj_in(x)
        out = self.attn(x, x, x)
        if isinstance(out, tuple):
            out = out[0]
        x = self.proj_out(out)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# SwinTransformerV2MSL — the full SkySense++ HR backbone
# ─────────────────────────────────────────────────────────────────────────────

class SwinTransformerV2MSL(SwinTransformerV2):
    """SwinV2-Huge with Multi-Scale Learning (annotation token fusion).

    For frozen feature extraction without annotations, pass dummy
    annotation images (e.g., zeros) to forward().
    """

    def __init__(self, **kwargs):
        if 'use_attn' in kwargs:
            self.use_attn = kwargs.pop('use_attn')
        else:
            self.use_attn = False
        if 'merge_stage' in kwargs:
            self.merge_stage = kwargs.pop('merge_stage')
        else:
            self.merge_stage = 0
        if 'with_cls_pos' in kwargs:
            self.with_cls_pos = kwargs.pop('with_cls_pos')
        else:
            self.with_cls_pos = False
        super().__init__(**kwargs)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        self.vocabulary_token = nn.Parameter(
            torch.zeros(self.vocabulary_size, self.embed_dims))
        self.vocabulary_weight = nn.Parameter(
            torch.zeros(1, self.patch_size * self.patch_size))
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.vocabulary_token, std=.02)

        if self.use_attn:
            self.attn1 = Proj_MHSA(embed_dims=352, proj_dims=256,
                                   num_heads=16, batch_first=True, bias=True)
            self.attn2 = Proj_MHSA(embed_dims=704, proj_dims=512,
                                   num_heads=16, batch_first=True, bias=True)
            self.attn3 = Proj_MHSA(embed_dims=1408, proj_dims=1024,
                                   num_heads=16, batch_first=True, bias=True)
            self.attention_blocks = [self.attn1, self.attn2, self.attn3]
            self.norm_attn = build_norm_layer(dict(type='LN'), 1408)[1]

    def create_ann_token(self, anno_img):
        B, H, W = anno_img.shape
        ann_token = torch.index_select(
            self.vocabulary_token, 0,
            anno_img.reshape(-1)).reshape(B, H, W, -1)
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        nph, npw = H // self.patch_size, W // self.patch_size
        weight = (F.softmax(self.vocabulary_weight, dim=1)
                  * self.patch_size * self.patch_size)
        weight = weight.reshape(
            1, 1, self.patch_size, 1, self.patch_size
        ).repeat(1, nph, 1, npw, 1).reshape(1, H, W, 1)
        ann_token = ann_token * weight
        ann_token = F.avg_pool2d(
            torch.einsum('BHWC->BCHW', ann_token),
            self.patch_size, self.patch_size)
        ann_token = torch.einsum('BCHW->BHWC', ann_token).reshape(
            B, nph * npw, self.embed_dims)
        return ann_token

    def forward(self, hr_img, anno_img=None, mask=None):
        """Forward pass.

        Args:
            hr_img: (B, 3, H, W) RGB image.
            anno_img: (B, H, W) integer annotation image, or None.
                If None, annotation tokens are skipped (zero contribution).
            mask: optional binary mask for pretraining. Ignored for inference.

        Returns:
            Tuple of feature maps from each output stage.
        """
        x, hw_shape = self.patch_embed(hr_img)

        # If no annotation image provided, skip MSL and use base SwinV2 path
        if anno_img is None:
            if self.use_abs_pos_embed:
                x = x + resize_pos_embed(
                    self.absolute_pos_embed, self.patch_resolution, hw_shape,
                    self.interpolate_mode, self.num_extra_tokens)
            x = self.drop_after_pos(x)

            outs = []
            for i, stage in enumerate(self.stages):
                x, hw_shape = stage(x, hw_shape)
                if self.use_attn and i <= len(self.attention_blocks) - 1:
                    x = x + self.attention_blocks[i](x)
                    if i == len(self.attention_blocks) - 1:
                        x = self.norm_attn(x)
                if i in self.out_indices:
                    norm_layer = getattr(self, f'norm{i}')
                    out = norm_layer(x)
                    out = out.view(-1, *hw_shape,
                                   stage.out_channels).permute(
                                       0, 3, 1, 2).contiguous()
                    outs.append(out)
            return tuple(outs)

        # Full MSL path with annotation tokens
        y = self.create_ann_token(anno_img)
        assert x.shape == y.shape
        B, L, C = y.shape
        if mask is not None:
            mask_tokens = self.mask_token.expand(B, L, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
            y = y * (1. - w) + mask_tokens * w

        if self.merge_stage == 0:
            x = (x + y) * 0.5
        else:
            x = x.reshape(B, *hw_shape, C)
            y = y.reshape(B, *hw_shape, C)
            x = torch.cat((x, y), dim=2)
            hw_shape = (hw_shape[0], hw_shape[1] * 2)
            x = x.reshape(B, -1, C)

        if self.use_abs_pos_embed:
            x = x + resize_pos_embed(
                self.absolute_pos_embed, self.patch_resolution, hw_shape,
                self.interpolate_mode, self.num_extra_tokens)
            if self.with_cls_pos:
                hw_shape_half = [hw_shape[0], hw_shape[1] // 2]
                x = x.reshape(B, *hw_shape, C)
                x1 = x[:, :, :x.shape[2]//2, :].reshape(B, -1, C)
                x2 = x[:, :, x.shape[2]//2:, :].reshape(B, -1, C)
                x1 = x1 + resize_pos_embed(
                    self.absolute_pos_embed, self.patch_resolution,
                    hw_shape_half, self.interpolate_mode,
                    self.num_extra_tokens)
                x2 = x2 + resize_pos_embed(
                    self.absolute_pos_embed, self.patch_resolution,
                    hw_shape_half, self.interpolate_mode,
                    self.num_extra_tokens)
                x1 = x1.reshape(B, *hw_shape_half, C)
                x2 = x2.reshape(B, *hw_shape_half, C)
                x = torch.cat((x1, x2), dim=2).reshape(B, -1, C)

        x = self.drop_after_pos(x)

        outs = []
        merge_idx = self.merge_stage - 1
        for i, stage in enumerate(self.stages):
            x, hw_shape = stage(x, hw_shape)
            if i == merge_idx:
                x = x.reshape(x.shape[0], *hw_shape, x.shape[-1])
                x = (x[:, :, :x.shape[2]//2]
                     + x[:, :, x.shape[2]//2:]) * 0.5
                x = x.reshape(x.shape[0], -1, x.shape[-1])
                hw_shape = (hw_shape[0], hw_shape[1] // 2)
            if self.use_attn:
                if i <= len(self.attention_blocks) - 1:
                    x = x + self.attention_blocks[i](x)
                if i == len(self.attention_blocks) - 1:
                    x = self.norm_attn(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape,
                               stage.out_channels).permute(
                                   0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

def build_skysense_hr_backbone(
    pretrained_path: str | None = None,
    img_size: int = 224,
    frozen: bool = True,
) -> SwinTransformerV2MSL:
    """Build a SkySense++ HR backbone (SwinV2-Huge) with optional pretrained
    weights.

    Args:
        pretrained_path: Path to the .pth weights file, or None.
        img_size: Input image size.
        frozen: If True, freeze all parameters after loading.

    Returns:
        Instantiated and optionally frozen SwinTransformerV2MSL model.
    """
    model = SwinTransformerV2MSL(
        arch='huge',
        img_size=img_size,
        patch_size=4,
        in_channels=3,
        vocabulary_size=64,
        window_size=8,
        drop_rate=0.,
        drop_path_rate=0.2,
        out_indices=(0, 1, 2, 3),
        use_abs_pos_embed=False,
        interpolate_mode='bicubic',
        with_cp=False,
        frozen_stages=-1,
        norm_eval=False,
        pad_small_map=True,
        pretrained_window_sizes=[0, 0, 0, 0],
        use_attn=True,
        merge_stage=2,
    )

    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[SkySense++] Loaded weights from {pretrained_path}")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    if frozen:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        print("[SkySense++] All parameters frozen.")

    return model
