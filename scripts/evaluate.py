"""
Generate paper figures and evaluation metrics for multiple models.

Supports any number of models — loads via build_model from train.py.

Generates:
  1. Qualitative comparison (RGB | GT | Model1 | Model2 | ...)
  2. Confusion matrices (one per model)
  3. Per-class IoU grouped bar chart
  4. Delta IoU heatmap (improvement over baseline)
  5. Full metrics table (OA, mIoU, mF1, Kappa, per-class IoU/F1)
  6. CSV export of all metrics

Usage:
  python scripts/evaluate.py \
    --configs configs/potsdam_baseline.yaml configs/potsdam_upanopticon.yaml \
    --ckpts outputs/unet_efficientnet-b4/best.pt outputs/u_panopticon_efficientnet-b4/best.pt \
    --labels "UNet-EffB4" "U-Panopticon" \
    --n 6 --out outputs/figures
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import yaml

# Use train.py's build_model for consistency
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train import build_model


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


def load_model_from_config(cfg, ckpt_path, device):
    model, variant_name, forward_fn, set_eval_backbones = build_model(
        cfg["model"], device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model, variant_name, forward_fn


def compute_full_metrics(preds, targets, ignore_index=0):
    valid = targets != ignore_index
    p, t = preds[valid], targets[valid]
    total = t.numel()
    oa = (p == t).float().mean().item()

    ious, f1s, precisions, recalls = [], [], [], []
    for c in range(1, len(CLASS_NAMES) + 1):
        pc, tc = p == c, t == c
        tp = (pc & tc).sum().item()
        fp = (pc & ~tc).sum().item()
        fn = (~pc & tc).sum().item()
        union = tp + fp + fn
        ious.append(tp / union if union > 0 else 0.0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)

    miou = np.mean([x for x in ious if x > 0])
    mf1 = np.mean(f1s)

    pe = sum(((p == c).sum().item() / total) * ((t == c).sum().item() / total)
             for c in range(1, len(CLASS_NAMES) + 1))
    kappa = (oa - pe) / (1.0 - pe) if (1.0 - pe) > 0 else 0.0

    return {
        'oa': oa, 'miou': miou, 'mf1': mf1, 'kappa': kappa,
        'ious': ious, 'f1s': f1s, 'precisions': precisions, 'recalls': recalls,
    }


# -- Figure 1: Qualitative comparison ----------------------------------------

def fig_qualitative(samples, labels, out_dir, n=6):
    num_models = len(labels)
    cols = 2 + num_models
    fig, axes = plt.subplots(n, cols, figsize=(4 * cols, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    titles = ["RGB", "Ground Truth"] + labels
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=12, fontweight='bold')

    for row, s in enumerate(samples[:n]):
        rgb = percentile_stretch(s['ms'][:3].permute(1, 2, 0).cpu().numpy())
        gt_rgb = gt_to_rgb(s['gt'].cpu().numpy())

        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(gt_rgb)

        for m, label in enumerate(labels):
            pred_rgb = gt_to_rgb(s['preds'][label].cpu().numpy())
            axes[row, 2 + m].imshow(pred_rgb)

        for col in range(cols):
            axes[row, col].axis('off')

    patches = [mpatches.Patch(color=np.array(CLASS_COLORS[i+1]) / 255.0,
                              label=CLASS_NAMES[i]) for i in range(6)]
    fig.legend(handles=patches, loc='lower center', ncol=6,
               fontsize=11, frameon=False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    path = os.path.join(out_dir, "qualitative_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# -- Figure 2: Confusion matrices --------------------------------------------

def compute_confusion_matrix(pred, target, ignore_index=0):
    n = len(CLASS_NAMES)
    valid = target != ignore_index
    p = pred[valid].numpy()
    t = target[valid].numpy()
    cm = np.zeros((n, n), dtype=np.int64)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            cm[i-1, j-1] = ((t == i) & (p == j)).sum()
    return cm


def fig_confusion_matrices(all_preds, all_targets, labels, out_dir,
                           ignore_index=0):
    num_models = len(labels)
    fig, axes = plt.subplots(1, num_models,
                              figsize=(7 * num_models, 6))
    if num_models == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        cm = compute_confusion_matrix(all_preds[label], all_targets,
                                       ignore_index)
        cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True).clip(1)
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(label, fontsize=13, fontweight='bold')
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

    fig.colorbar(im, ax=list(axes), shrink=0.6, label="Recall")
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrices.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# -- Figure 3: Per-class IoU bar chart ---------------------------------------

def fig_iou_comparison(all_metrics, labels, out_dir):
    x = np.arange(len(CLASS_NAMES))
    num_models = len(labels)
    width = 0.8 / num_models

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0', '#00BCD4']

    fig, ax = plt.subplots(figsize=(12, 5))
    for m, label in enumerate(labels):
        ious = all_metrics[label]['ious']
        miou = all_metrics[label]['miou']
        offset = (m - num_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, ious, width,
                      label=f'{label} (mIoU={miou:.3f})',
                      color=colors[m % len(colors)], edgecolor='white')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.2f}', ha='center', va='bottom',
                    fontsize=7)

    ax.set_ylabel('IoU', fontsize=12)
    ax.set_title('Per-Class IoU Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "iou_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# -- Figure 4: Delta IoU heatmap ---------------------------------------------

def fig_delta_iou(all_metrics, labels, out_dir):
    if len(labels) < 2:
        return
    baseline_label = labels[0]
    baseline_ious = np.array(all_metrics[baseline_label]['ious'])

    model_labels = labels[1:]
    deltas = []
    for label in model_labels:
        ious = np.array(all_metrics[label]['ious'])
        deltas.append(ious - baseline_ious)
    deltas = np.array(deltas)

    fig, ax = plt.subplots(figsize=(10, max(3, len(model_labels) * 1.2)))
    vmax = max(abs(deltas.min()), abs(deltas.max()), 0.05)
    im = ax.imshow(deltas, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=10)
    ax.set_title(f'IoU Improvement over {baseline_label}', fontsize=13,
                 fontweight='bold')

    for i in range(len(model_labels)):
        for j in range(len(CLASS_NAMES)):
            val = deltas[i, j]
            color = 'black' if abs(val) < vmax * 0.6 else 'white'
            ax.text(j, i, f'{val:+.2f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    fig.colorbar(im, ax=ax, shrink=0.8, label="Delta IoU")
    plt.tight_layout()
    path = os.path.join(out_dir, "delta_iou_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models and generate paper figures.")
    parser.add_argument('--configs', nargs='+', required=True,
                        help='Config files for each model')
    parser.add_argument('--ckpts', nargs='+', required=True,
                        help='Checkpoint files for each model')
    parser.add_argument('--labels', nargs='+', required=True,
                        help='Display labels for each model')
    parser.add_argument('--n', type=int, default=6,
                        help='Number of qualitative samples')
    parser.add_argument('--out', type=str, default='outputs/figures')
    args = parser.parse_args()

    assert len(args.configs) == len(args.ckpts) == len(args.labels), \
        "Must provide same number of configs, ckpts, and labels"

    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load first config for dataset (shared across all models)
    with open(args.configs[0]) as f:
        data_cfg = yaml.safe_load(f)

    ignore_index = data_cfg["dataset"].get("ignore_index", 0)
    wavelengths = data_cfg["dataset"]["bands"]["wavelengths_nm"]
    rgb_indices = tuple(data_cfg["dataset"]["bands"]["rgb_indices"])

    # Load dataset
    from core.datasets.potsdam import PotsdamDataModule
    dm = PotsdamDataModule(data_cfg)
    dm.setup()
    test_loader = dm.test_loader()

    # Load all models
    models = {}
    forward_fns = {}
    for config_path, ckpt_path, label in zip(args.configs, args.ckpts, args.labels):
        print(f"Loading {label}...")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        model, variant, forward_fn = load_model_from_config(cfg, ckpt_path, device)
        models[label] = model
        forward_fns[label] = forward_fn

    # Pre-allocate chn_ids
    max_bs = data_cfg["dataset"]["batch_size"]
    chn_ids_base = torch.tensor(
        [wavelengths] * max_bs, dtype=torch.float32, device=device)

    # -- Run inference -------------------------------------------------------
    print(f"\nRunning inference on test set ({len(test_loader)} batches)...")
    all_preds = {label: [] for label in args.labels}
    all_targets = []
    qualitative_samples = []

    with torch.no_grad():
        for i, (ms, ndsm, gt) in enumerate(test_loader):
            ms, ndsm, gt = ms.to(device), ndsm.to(device), gt.to(device)
            ms, ndsm = dm.normalize(ms, ndsm, device=device)

            B = ms.shape[0]
            chn_ids = chn_ids_base[:B]

            sample_preds = {}
            for label in args.labels:
                out = forward_fns[label](models[label], ms, ndsm, chn_ids,
                                         rgb_indices)
                pred = out['logits'].argmax(dim=1)
                all_preds[label].append(pred.cpu())
                sample_preds[label] = pred

            all_targets.append(gt.cpu())

            # Collect qualitative samples
            if len(qualitative_samples) < args.n:
                for b in range(ms.shape[0]):
                    if len(qualitative_samples) >= args.n:
                        break
                    qualitative_samples.append({
                        'ms': ms[b].cpu() * dm.ms_std.view(-1, 1, 1)
                              + dm.ms_mean.view(-1, 1, 1),
                        'gt': gt[b].cpu(),
                        'preds': {label: sample_preds[label][b].cpu()
                                  for label in args.labels},
                    })

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(test_loader)} batches...")

    # Concatenate
    for label in args.labels:
        all_preds[label] = torch.cat(all_preds[label])
    all_targets = torch.cat(all_targets)

    # -- Compute metrics -----------------------------------------------------
    print("\nComputing metrics...")
    all_metrics = {}
    for label in args.labels:
        all_metrics[label] = compute_full_metrics(
            all_preds[label], all_targets, ignore_index)

    # -- Generate figures ----------------------------------------------------
    print("\nGenerating figures...")

    print("  [1/4] Qualitative comparison...")
    fig_qualitative(qualitative_samples, args.labels, args.out, n=args.n)

    print("  [2/4] Confusion matrices...")
    fig_confusion_matrices(all_preds, all_targets, args.labels, args.out,
                           ignore_index)

    print("  [3/4] Per-class IoU comparison...")
    fig_iou_comparison(all_metrics, args.labels, args.out)

    print("  [4/4] Delta IoU heatmap...")
    fig_delta_iou(all_metrics, args.labels, args.out)

    # -- Print summary table -------------------------------------------------
    header = f"  {'Metric':<12}"
    for label in args.labels:
        header += f" {label:>16}"
    print(f"\n{'=' * len(header)}")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for metric_name, key in [('OA', 'oa'), ('mIoU', 'miou'),
                              ('mF1', 'mf1'), ('Kappa', 'kappa')]:
        row = f"  {metric_name:<12}"
        for label in args.labels:
            row += f" {all_metrics[label][key]:>16.4f}"
        print(row)

    print(f"  {'-' * (len(header) - 2)}")
    print(f"  {'Class':<12}", end="")
    for label in args.labels:
        print(f" {'IoU':>8}{'F1':>8}", end="")
    print()

    for i, name in enumerate(CLASS_NAMES):
        row = f"  {name:<12}"
        for label in args.labels:
            row += f" {all_metrics[label]['ious'][i]:>8.4f}"
            row += f"{all_metrics[label]['f1s'][i]:>8.4f}"
        print(row)

    print(f"{'=' * len(header)}")

    # Save metrics as CSV
    csv_path = os.path.join(args.out, "metrics.csv")
    with open(csv_path, 'w') as f:
        f.write("Model,OA,mIoU,mF1,Kappa," +
                ",".join(f"IoU_{c}" for c in CLASS_NAMES) + "," +
                ",".join(f"F1_{c}" for c in CLASS_NAMES) + "\n")
        for label in args.labels:
            m = all_metrics[label]
            f.write(f"{label},{m['oa']:.4f},{m['miou']:.4f},"
                    f"{m['mf1']:.4f},{m['kappa']:.4f},")
            f.write(",".join(f"{x:.4f}" for x in m['ious']))
            f.write(",")
            f.write(",".join(f"{x:.4f}" for x in m['f1s']))
            f.write("\n")
    print(f"\nMetrics saved to: {csv_path}")
    print(f"Figures saved to: {args.out}/")


if __name__ == '__main__':
    main()
