"""Visualize random Potsdam samples: RGB, IR, nDSM, ground truth side-by-side."""

import argparse
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch


CLASS_COLORS = np.array([
    [0, 0, 0],         # 0: ignore
    [255, 0, 0],       # 1: Clutter/background
    [180, 180, 180],   # 2: Impervious surfaces
    [0, 0, 255],       # 3: Building
    [0, 255, 255],     # 4: Low vegetation
    [0, 255, 0],       # 5: Tree
    [255, 255, 0],     # 6: Car
], dtype=np.uint8)

CLASS_NAMES = [
    "Ignore", "Clutter", "Impervious", "Building",
    "Low veg", "Tree", "Car"
]


def percentile_stretch(img, lo=2, hi=98):
    lo_v = np.percentile(img, lo, axis=(0, 1), keepdims=True)
    hi_v = np.percentile(img, hi, axis=(0, 1), keepdims=True)
    return np.clip((img - lo_v) / (hi_v - lo_v + 1e-6), 0, 1)


def gt_to_rgb(gt):
    h, w = gt.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(CLASS_COLORS)):
        rgb[gt == c] = CLASS_COLORS[c]
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/potsdam.yaml')
    parser.add_argument('--n', type=int, default=4)
    parser.add_argument('--out', type=str, default='outputs/samples.png')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from core.datasets.potsdam import PotsdamDataModule

    print("Setting up dataset...")
    dm = PotsdamDataModule(cfg)
    dm.setup()

    train_ds = dm.train_dataset
    indices = random.sample(range(len(train_ds)), min(args.n, len(train_ds)))

    fig, axes = plt.subplots(args.n, 4, figsize=(16, 4 * args.n))
    if args.n == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        sample = train_ds[idx]
        ms = sample["ms"].numpy()       # (4, H, W)
        ndsm = sample["ndsm"].numpy()   # (1, H, W)
        gt = sample["gt"].numpy()       # (H, W)
        tile = sample["tile_id"]

        # RGB (bands 0,1,2)
        rgb = percentile_stretch(ms[:3].transpose(1, 2, 0))
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"RGB ({tile})")
        axes[row, 0].axis('off')

        # IR (band 3)
        ir = percentile_stretch(ms[3])
        axes[row, 1].imshow(ir, cmap='RdYlGn')
        axes[row, 1].set_title("IR")
        axes[row, 1].axis('off')

        # nDSM
        axes[row, 2].imshow(ndsm[0], cmap='viridis',
                            vmin=0, vmax=np.percentile(ndsm[0], 98))
        axes[row, 2].set_title(f"nDSM (max={ndsm.max():.1f}m)")
        axes[row, 2].axis('off')

        # Ground truth
        gt_rgb = gt_to_rgb(gt)
        axes[row, 3].imshow(gt_rgb)
        axes[row, 3].set_title("Ground Truth")
        axes[row, 3].axis('off')

    # Legend
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, fc=np.array(CLASS_COLORS[i]) / 255.0)
        for i in range(1, len(CLASS_NAMES))
    ]
    fig.legend(legend_patches, CLASS_NAMES[1:], loc='lower center',
               ncol=6, fontsize=10, frameon=False)

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved to {args.out}")
    plt.close()


if __name__ == '__main__':
    main()
