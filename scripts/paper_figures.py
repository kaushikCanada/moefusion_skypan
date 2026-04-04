"""
Generate paper figures comparing MoE fusion vs baseline.

  1. Qualitative comparison (RGB | GT | MoE | Baseline) — N random test patches
  2. Confusion matrices for both models
  3. Gate weight distribution (histogram + per-scene scatter)
  4. Per-class IoU bar chart comparison

Usage:
  python scripts/paper_figures.py \
    --moe-config configs/potsdam_moe.yaml \
    --moe-ckpt outputs/moe_fusion/best.pt \
    --baseline-config configs/potsdam_baseline.yaml \
    --baseline-ckpt outputs/baseline/best.pt \
    --n 6 --out outputs/figures
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import yaml


CLASS_COLORS = np.array([
    [0, 0, 0],         # 0: ignore
    [255, 0, 0],       # 1: Clutter
    [180, 180, 180],   # 2: Impervious
    [0, 0, 255],       # 3: Building
    [0, 255, 255],     # 4: Low veg
    [0, 255, 0],       # 5: Tree
    [255, 255, 0],     # 6: Car
], dtype=np.uint8)

CLASS_NAMES = ["Clutter", "Impervious", "Building", "Low veg", "Tree", "Car"]


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


def load_model(model_type, cfg, ckpt_path, device):
    model_cfg = cfg["model"]
    if model_type == 'moe':
        from core.models.moe_segmentor import MoESegmentor
        model = MoESegmentor(
            num_classes=model_cfg["num_classes"],
            skysense_weights=model_cfg.get("skysense_weights"),
            img_size=model_cfg.get("img_size", 256),
            panopticon_checkpoint=model_cfg.get("panopticon_checkpoint"),
            fpn_channels=model_cfg.get("fpn_channels", 256),
            dropout=0.0,
        )
    else:
        from core.models.baseline_unet import BaselineUNet
        model = BaselineUNet(
            num_classes=model_cfg["num_classes"],
            in_channels=model_cfg.get("in_channels", 5),
            encoder_name=model_cfg.get("encoder_name", "efficientnet-b4"),
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device).eval()
    return model


def predict_moe(model, ms, ndsm, wavelengths, rgb_indices, device):
    B = ms.shape[0]
    chn_ids = torch.tensor(
        [wavelengths] * B, dtype=torch.float32, device=device)
    out = model(ms, chn_ids, ndsm, rgb_indices=rgb_indices)
    return out['logits'].argmax(dim=1), out['gate_weights']


def predict_baseline(model, ms, ndsm):
    x = torch.cat([ms, ndsm], dim=1)
    return model(x).argmax(dim=1), None


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Qualitative comparison
# ─────────────────────────────────────────────────────────────────────────────

def fig_qualitative(samples, out_dir, n=6):
    """RGB | GT | MoE pred | Baseline pred — N rows."""
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    titles = ["RGB", "Ground Truth", "MoE Fusion (Ours)", "UNet-EffB4 Baseline"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=13, fontweight='bold')

    for row, s in enumerate(samples[:n]):
        rgb = percentile_stretch(s['ms'][:3].permute(1, 2, 0).cpu().numpy())
        gt_rgb = gt_to_rgb(s['gt'].cpu().numpy())
        moe_rgb = gt_to_rgb(s['moe_pred'].cpu().numpy())
        base_rgb = gt_to_rgb(s['base_pred'].cpu().numpy())

        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(gt_rgb)
        axes[row, 2].imshow(moe_rgb)
        axes[row, 3].imshow(base_rgb)

        for col in range(4):
            axes[row, col].axis('off')

    # Legend
    patches = [mpatches.Patch(color=np.array(CLASS_COLORS[i+1]) / 255.0,
                              label=CLASS_NAMES[i]) for i in range(6)]
    fig.legend(handles=patches, loc='lower center', ncol=6,
               fontsize=11, frameon=False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    path = os.path.join(out_dir, "qualitative_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Confusion matrices
# ─────────────────────────────────────────────────────────────────────────────

def compute_confusion_matrix(pred, target, num_classes, ignore_index=0):
    """Confusion matrix for classes 1..6."""
    n = len(CLASS_NAMES)  # 6
    valid = target != ignore_index
    p = pred[valid].numpy()
    t = target[valid].numpy()
    cm = np.zeros((n, n), dtype=np.int64)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            cm[i-1, j-1] = ((t == i) & (p == j)).sum()
    return cm


def fig_confusion_matrices(moe_preds, moe_targets, base_preds, base_targets,
                           num_classes, out_dir, ignore_index=0):
    cm_moe = compute_confusion_matrix(moe_preds, moe_targets, num_classes, ignore_index)
    cm_base = compute_confusion_matrix(base_preds, base_targets, num_classes, ignore_index)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for ax, cm, title in [(ax1, cm_moe, "MoE Fusion (Ours)"),
                           (ax2, cm_base, "UNet-EffB4 Baseline")]:
        # Normalize per row
        cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True).clip(1)
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(6):
            for j in range(6):
                val = cm_norm[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        fontsize=8, color=color)

    fig.colorbar(im, ax=[ax1, ax2], shrink=0.6, label="Recall")
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrices.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Gate weight visualization
# ─────────────────────────────────────────────────────────────────────────────

def fig_gate_weights(all_gate_weights, out_dir):
    """Histogram of gate weights per scale + scatter of w0 vs w2."""
    gw = np.array(all_gate_weights)  # (N, 4)
    scale_names = ["Stage 0\n(1/4, fine)", "Stage 1\n(1/8, mid)",
                   "Stage 2\n(1/16, semantic)", "Stage 3\n(1/32, global)"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    for i in range(4):
        ax.hist(gw[:, i], bins=30, alpha=0.6, label=scale_names[i],
                color=colors[i])
    ax.axvline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel("Gate Weight", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Scale Gate Weight Distribution", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    # Scatter: w1 (suppressed) vs w2 (boosted)
    ax = axes[1]
    sc = ax.scatter(gw[:, 1], gw[:, 2], c=gw[:, 0], cmap='viridis',
                    alpha=0.5, s=10)
    ax.set_xlabel("w1 (Stage 1, mid-level)", fontsize=11)
    ax.set_ylabel("w2 (Stage 2, semantic)", fontsize=11)
    ax.set_title("Gate Routing: Stage 1 vs Stage 2", fontsize=13, fontweight='bold')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(1.0, color='gray', linestyle='--', linewidth=0.5)
    plt.colorbar(sc, ax=ax, label="w0 (fine detail)")

    plt.tight_layout()
    path = os.path.join(out_dir, "gate_weights.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Per-class IoU bar chart
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_class_iou(pred, target, num_classes, ignore_index=0):
    """Compute IoU for classes 1..6 (skip ignore_index=0)."""
    ious = []
    for c in range(1, len(CLASS_NAMES) + 1):  # 1..6
        pred_c = pred == c
        target_c = target == c
        tp = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        ious.append(tp / union if union > 0 else 0.0)
    return ious


def fig_iou_comparison(moe_ious, base_ious, moe_miou, base_miou, out_dir):
    x = np.arange(len(CLASS_NAMES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, moe_ious, width, label=f'MoE Fusion (mIoU={moe_miou:.3f})',
                   color='#2196F3', edgecolor='white')
    bars2 = ax.bar(x + width/2, base_ious, width, label=f'UNet-EffB4 (mIoU={base_miou:.3f})',
                   color='#FF9800', edgecolor='white')

    ax.set_ylabel('IoU', fontsize=12)
    ax.set_title('Per-Class IoU Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "iou_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--moe-config', type=str, required=True)
    parser.add_argument('--moe-ckpt', type=str, required=True)
    parser.add_argument('--baseline-config', type=str, required=True)
    parser.add_argument('--baseline-ckpt', type=str, required=True)
    parser.add_argument('--n', type=int, default=6,
                        help='Number of qualitative samples')
    parser.add_argument('--out', type=str, default='outputs/figures')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load configs
    with open(args.moe_config) as f:
        moe_cfg = yaml.safe_load(f)
    with open(args.baseline_config) as f:
        base_cfg = yaml.safe_load(f)

    ignore_index = moe_cfg["dataset"].get("ignore_index", 0)
    num_classes = moe_cfg["model"]["num_classes"]
    wavelengths = moe_cfg["dataset"]["bands"]["wavelengths_nm"]
    rgb_indices = tuple(moe_cfg["dataset"]["bands"]["rgb_indices"])

    # Load dataset (shared — same splits)
    from core.datasets.potsdam import PotsdamDataModule
    dm = PotsdamDataModule(moe_cfg)
    dm.setup()
    test_loader = dm.test_loader()

    # Load models
    print("Loading models...")
    moe_model = load_model('moe', moe_cfg, args.moe_ckpt, device)
    base_model = load_model('baseline', base_cfg, args.baseline_ckpt, device)

    # ── Run inference on full test set ──────────────────────────────────
    print("Running inference on test set...")
    all_moe_preds, all_base_preds, all_targets = [], [], []
    all_gate_weights = []
    qualitative_samples = []

    with torch.no_grad():
        for i, (ms, ndsm, gt) in enumerate(test_loader):
            ms, ndsm, gt = ms.to(device), ndsm.to(device), gt.to(device)
            ms, ndsm = dm.normalize(ms, ndsm, device=device)

            moe_pred, gw = predict_moe(
                moe_model, ms, ndsm, wavelengths, rgb_indices, device)
            base_pred, _ = predict_baseline(base_model, ms, ndsm)

            all_moe_preds.append(moe_pred.cpu())
            all_base_preds.append(base_pred.cpu())
            all_targets.append(gt.cpu())
            all_gate_weights.extend(gw.cpu().numpy().tolist())

            # Collect qualitative samples (unnormalized for display)
            if len(qualitative_samples) < args.n:
                for b in range(ms.shape[0]):
                    if len(qualitative_samples) >= args.n:
                        break
                    qualitative_samples.append({
                        'ms': ms[b].cpu() * dm.ms_std.view(-1, 1, 1)
                              + dm.ms_mean.view(-1, 1, 1),
                        'gt': gt[b].cpu(),
                        'moe_pred': moe_pred[b].cpu(),
                        'base_pred': base_pred[b].cpu(),
                    })

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(test_loader)} batches...")

    all_moe_preds = torch.cat(all_moe_preds)
    all_base_preds = torch.cat(all_base_preds)
    all_targets = torch.cat(all_targets)

    # ── Generate figures ────────────────────────────────────────────────
    print("\nGenerating figures...")

    # 1. Qualitative
    print("  [1/4] Qualitative comparison...")
    fig_qualitative(qualitative_samples, args.out, n=args.n)

    # 2. Confusion matrices
    print("  [2/4] Confusion matrices...")
    fig_confusion_matrices(all_moe_preds, all_targets,
                          all_base_preds, all_targets,
                          num_classes, args.out, ignore_index)

    # 3. Gate weights
    print("  [3/4] Gate weight visualization...")
    fig_gate_weights(all_gate_weights, args.out)

    # 4. IoU bar chart
    print("  [4/4] Per-class IoU comparison...")
    moe_ious = compute_per_class_iou(all_moe_preds, all_targets,
                                      num_classes, ignore_index)
    base_ious = compute_per_class_iou(all_base_preds, all_targets,
                                       num_classes, ignore_index)
    moe_miou = np.mean([x for x in moe_ious if x > 0])
    base_miou = np.mean([x for x in base_ious if x > 0])
    fig_iou_comparison(moe_ious, base_ious, moe_miou, base_miou, args.out)

    # ── Print summary table ─────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  {'Metric':<12} {'MoE Fusion':>12} {'UNet-EffB4':>12}")
    print(f"  {'-'*36}")
    print(f"  {'mIoU':<12} {moe_miou:>12.4f} {base_miou:>12.4f}")

    valid = all_targets != ignore_index
    moe_oa = (all_moe_preds[valid] == all_targets[valid]).float().mean().item()
    base_oa = (all_base_preds[valid] == all_targets[valid]).float().mean().item()
    print(f"  {'OA':<12} {moe_oa:>12.4f} {base_oa:>12.4f}")

    print(f"  {'-'*36}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<12} {moe_ious[i]:>12.4f} {base_ious[i]:>12.4f}")
    print(f"{'='*50}")

    print(f"\nAll figures saved to: {args.out}/")


if __name__ == '__main__':
    main()
