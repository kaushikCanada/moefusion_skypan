"""
Generate qualitative comparison figure only (no metrics computation).

Runs inference on N random test patches and shows:
  RGB | Ground Truth | Model1 | Model2 | ...

Usage:
  python scripts/plot_qualitative.py \
    --configs configs/potsdam_baseline.yaml configs/potsdam_upanopticon.yaml \
    --ckpts outputs/unet_efficientnet-b4/best.pt outputs/u_panopticon_efficientnet-b4/best.pt \
    --labels "UNet-EffB4" "U-Panopticon" \
    --n 6 --seed 42 --out outputs/figures/qualitative_comparison.png
"""

import argparse
import os
import sys
import random

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train import build_model


CLASS_COLORS = np.array([
    [0, 0, 0],         # 0: ignore
    [180, 180, 180],   # 1: Impervious
    [0, 0, 255],       # 2: Building
    [0, 255, 255],     # 3: Low veg
    [0, 255, 0],       # 4: Tree
    [255, 255, 0],     # 5: Car
], dtype=np.uint8)

CLASS_NAMES = ["Impervious", "Building", "Low veg", "Tree", "Car"]


def gt_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(CLASS_COLORS)):
        rgb[mask == c] = CLASS_COLORS[c]
    return rgb


def percentile_stretch(img, lo=2, hi=98):
    lo_v = np.percentile(img, lo, axis=(0, 1), keepdims=True)
    hi_v = np.percentile(img, hi, axis=(0, 1), keepdims=True)
    return np.clip((img - lo_v) / (hi_v - lo_v + 1e-6), 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-config', type=str,
                        default='configs/evaluate.yaml',
                        help='Path to evaluate.yaml')
    parser.add_argument('--n', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str,
                        default='outputs/figures/qualitative_comparison.png')
    args = parser.parse_args()

    # Read models from evaluate.yaml
    with open(args.eval_config) as f:
        eval_cfg = yaml.safe_load(f)
    args.configs = [m['config'] for m in eval_cfg['models']]
    args.ckpts = [m['checkpoint'] for m in eval_cfg['models']]
    args.labels = [m['label'] for m in eval_cfg['models']]
    dataset_config = eval_cfg.get('dataset_config', args.configs[0])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset from dataset_config
    with open(dataset_config) as f:
        data_cfg = yaml.safe_load(f)

    wavelengths = data_cfg["dataset"]["bands"]["wavelengths_nm"]
    max_bs = data_cfg["dataset"]["batch_size"]

    from core.datasets.potsdam import PotsdamDataModule
    dm = PotsdamDataModule(data_cfg)
    dm.setup()
    test_loader = dm.test_loader()

    # Load models
    models = {}
    forward_fns = {}
    for config_path, ckpt_path, label in zip(args.configs, args.ckpts, args.labels):
        print(f"Loading {label}...")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        model, _, forward_fn, _ = build_model(cfg["model"], device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        model.eval()
        models[label] = model
        forward_fns[label] = forward_fn

    chn_ids_base = torch.tensor(
        [wavelengths] * max_bs, dtype=torch.float32, device=device)
    rgb_indices = tuple(data_cfg["dataset"]["bands"]["rgb_indices"])

    # Pick N random batch indices based on seed
    random.seed(args.seed)
    num_batches = len(test_loader)
    batch_indices = sorted(random.sample(range(num_batches),
                                          min(args.n, num_batches)))

    # Run inference only on selected batches
    print(f"Collecting {args.n} samples (seed={args.seed})...")
    samples = []
    with torch.no_grad():
        for batch_idx, (ms, ndsm, ndvi, gt) in enumerate(test_loader):
            if batch_idx not in batch_indices:
                continue

            ms, ndsm, ndvi, gt = ms.to(device), ndsm.to(device), ndvi.to(device), gt.to(device)
            ms, ndsm = dm.normalize(ms, ndsm, device=device)

            B = ms.shape[0]
            chn_ids = chn_ids_base[:B]

            preds = {}
            for label in args.labels:
                out = forward_fns[label](models[label], ms, ndsm, ndvi, chn_ids,
                                         rgb_indices)
                pred = out['logits'].argmax(dim=1)
                # Mask predictions to 0 where GT is ignore
                pred[gt == 0] = 0
                preds[label] = pred

            # Take first sample from each selected batch
            # Denormalize nDSM for display
            ndsm_denorm = ndsm[0].cpu() * dm.ndsm_std.view(-1, 1, 1) \
                          + dm.ndsm_mean.view(-1, 1, 1)
            samples.append({
                'ms': ms[0].cpu() * dm.ms_std.view(-1, 1, 1)
                      + dm.ms_mean.view(-1, 1, 1),
                'ndsm': ndsm_denorm,
                'gt': gt[0].cpu(),
                'preds': {label: preds[label][0].cpu()
                          for label in args.labels},
            })

            if len(samples) >= args.n:
                break

    # Plot
    n = len(samples)
    cols = 3 + len(args.labels)  # RGB, nDSM, GT, models...
    fig, axes = plt.subplots(n, cols, figsize=(4 * cols, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    titles = ["RGB", "nDSM", "Ground Truth"] + args.labels
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=18, fontweight='bold')

    for row, s in enumerate(samples):
        rgb = percentile_stretch(s['ms'][:3].permute(1, 2, 0).cpu().numpy())
        ndsm_img = s['ndsm'].squeeze(0).cpu().numpy()
        gt_rgb = gt_to_rgb(s['gt'].cpu().numpy())

        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(ndsm_img, cmap='terrain')
        axes[row, 2].imshow(gt_rgb)

        for m, label in enumerate(args.labels):
            pred_rgb = gt_to_rgb(s['preds'][label].cpu().numpy())
            axes[row, 3 + m].imshow(pred_rgb)

        for col in range(cols):
            axes[row, col].axis('off')

    patches = [mpatches.Patch(color=[0, 0, 0], label="Background")]
    patches += [mpatches.Patch(color=np.array(CLASS_COLORS[i+1]) / 255.0,
                               label=CLASS_NAMES[i]) for i in range(5)]
    fig.legend(handles=patches, loc='lower center', ncol=6,
               fontsize=30, frameon=False)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {args.out}")


if __name__ == '__main__':
    main()
