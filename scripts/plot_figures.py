"""
Generate paper figures from metrics.csv (no inference needed).

Generates:
  1. Per-class IoU grouped bar chart
  2. Delta IoU heatmap (improvement over first model)
  3. Summary table printed to terminal

Usage:
  python scripts/plot_figures.py --csv outputs/figures/metrics.csv --out outputs/figures
"""

import argparse
import os
import csv

import matplotlib.pyplot as plt
import numpy as np


def read_metrics(csv_path):
    models = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            models.append(row)
    return models


def get_class_names(models):
    iou_keys = [k for k in models[0].keys() if k.startswith('IoU_')]
    return [k.replace('IoU_', '') for k in iou_keys]


def fig_iou_comparison(models, class_names, out_dir):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # ── Style ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':       'DejaVu Sans',
        'font.size':         20,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.spines.left':  False,
        'axes.spines.bottom':False,
    })

    # Paired colors: baseline muted, +Panopticon saturated
    # Gray=LinearProbe, Blue pair=UNet, Teal pair=SegFormer
    colors = [
        '#9E9E9E',   # Linear Probe — neutral gray
        '#90CAF9',   # UNet-EffB4   — light blue
        '#1565C0',   # U-Panopticon — deep blue
        '#80CBC4',   # SegFormer-B2 — light teal
        '#00695C',   # SegFormer+Pan — deep teal
    ]

    x          = np.arange(len(class_names))
    n          = len(models)
    width      = 0.14
    group_gap  = 0.04
    total_w    = n * width + (n - 1) * group_gap
    offsets    = np.linspace(-total_w / 2 + width / 2,
                              total_w / 2 - width / 2, n)

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')

    # ── Bars ─────────────────────────────────────────────────────────────────
    for m, (model, color) in enumerate(zip(models, colors)):
        ious   = [float(model[f'IoU_{c}']) for c in class_names]
        miou   = float(model['mIoU'])
        label  = model['Model']

        bars = ax.bar(
            x + offsets[m], ious, width,
            color=color,
            edgecolor='white',
            linewidth=0.6,
            zorder=3,
        )

        # Value labels — only on top two performers to avoid clutter
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.008,
                f'{h:.2f}',
                ha='center', va='bottom',
                fontsize=20,
                color='#444444',
            )

    # ── Grid ─────────────────────────────────────────────────────────────────
    ax.yaxis.grid(True, color='#E0E0E0', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_ylabel('IoU', fontsize=20, labelpad=8, color='#333333')
    ax.set_ylim(0, 1.10)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels([f'{v:.1f}' for v in np.arange(0, 1.1, 0.2)],
                       fontsize=20, color='#555555')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=20, color='#333333')
    ax.tick_params(axis='both', length=0)

    # ── Legend ───────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=colors[m],
                       label=f"{model['Model']} (mIoU={float(model['mIoU']):.3f})")
        for m, model in enumerate(models)
    ]
    ax.legend(
        handles=patches,
        fontsize=20,
        loc='lower right',
        frameon=True,
        framealpha=0.92,
        edgecolor='#CCCCCC',
        ncol=1,
    )

    # ── Separator lines between class groups ─────────────────────────────────
    for xi in x[:-1]:
        ax.axvline(xi + 0.5, color='#E0E0E0', linewidth=1.0,
                   linestyle='--', zorder=1)

    plt.tight_layout(pad=1.5)
    path = os.path.join(out_dir, "iou_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")

# def fig_iou_comparison(models, class_names, out_dir):
#     x = np.arange(len(class_names))
#     num_models = len(models)
#     width = 0.8 / num_models
#     colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0', '#00BCD4']

#     fig, ax = plt.subplots(figsize=(12, 5))
#     for m, model in enumerate(models):
#         ious = [float(model[f'IoU_{c}']) for c in class_names]
#         miou = float(model['mIoU'])
#         label = model['Model']
#         offset = (m - num_models / 2 + 0.5) * width
#         bars = ax.bar(x + offset, ious, width,
#                       label=f'{label} (mIoU={miou:.3f})',
#                       color=colors[m % len(colors)], edgecolor='white')
#         for bar in bars:
#             ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
#                     f'{bar.get_height():.2f}', ha='center', va='bottom',
#                     fontsize=7)

#     ax.set_ylabel('IoU', fontsize=12)
#     ax.set_title('Per-Class IoU Comparison', fontsize=14, fontweight='bold')
#     ax.set_xticks(x)
#     ax.set_xticklabels(class_names, fontsize=10)
#     ax.legend(fontsize=9, loc='lower right')
#     ax.set_ylim(0, 1.05)
#     ax.grid(axis='y', alpha=0.3)

#     plt.tight_layout()
#     path = os.path.join(out_dir, "iou_comparison.png")
#     fig.savefig(path, dpi=200, bbox_inches='tight')
#     plt.close()
#     print(f"  Saved: {path}")


def fig_delta_iou(models, class_names, out_dir):
    if len(models) < 2:
        return
    baseline = models[0]
    baseline_ious = np.array([float(baseline[f'IoU_{c}']) for c in class_names])

    model_labels = [m['Model'] for m in models[1:]]
    deltas = []
    for model in models[1:]:
        ious = np.array([float(model[f'IoU_{c}']) for c in class_names])
        deltas.append(ious - baseline_ious)
    deltas = np.array(deltas)

    fig, ax = plt.subplots(figsize=(10, max(3, len(model_labels) * 1.2)))
    vmax = max(abs(deltas.min()), abs(deltas.max()), 0.05)
    im = ax.imshow(deltas, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=10)
    ax.set_title(f'IoU Improvement over {baseline["Model"]}',
                 fontsize=13, fontweight='bold')

    for i in range(len(model_labels)):
        for j in range(len(class_names)):
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


def fig_summary_table(models, class_names, out_dir):
    fig, ax = plt.subplots(figsize=(14, max(3, len(models) * 0.6 + 2)))
    ax.axis('off')

    headers = ['Model', 'OA', 'mIoU', 'mF1', 'Kappa'] + class_names
    cell_data = []
    for model in models:
        row = [model['Model'],
               f"{float(model['OA']):.4f}",
               f"{float(model['mIoU']):.4f}",
               f"{float(model['mF1']):.4f}",
               f"{float(model['Kappa']):.4f}"]
        for c in class_names:
            row.append(f"{float(model[f'IoU_{c}']):.4f}")
        cell_data.append(row)

    table = ax.table(cellText=cell_data, colLabels=headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Bold header
    for j in range(len(headers)):
        table[0, j].set_text_props(fontweight='bold')
        table[0, j].set_facecolor('#E0E0E0')

    # Highlight best mIoU
    mious = [float(m['mIoU']) for m in models]
    best_idx = np.argmax(mious)
    for j in range(len(headers)):
        table[best_idx + 1, j].set_facecolor('#C8E6C9')

    plt.tight_layout()
    path = os.path.join(out_dir, "results_table.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig_confusion_matrices(npz_path, out_dir):
    data = np.load(npz_path, allow_pickle=True)
    labels = list(data['labels'])
    class_names = list(data['class_names'])
    num_models = len(labels)

    fig, all_axes = plt.subplots(1, num_models + 1,
                                  figsize=(7 * num_models + 1, 6),
                                  gridspec_kw={'width_ratios': [1] * num_models + [0.05]})
    axes = all_axes[:-1]
    cbar_ax = all_axes[-1]

    if num_models == 1:
        axes = [axes]

    for idx, (ax, label) in enumerate(zip(axes, labels)):
        cm = data[f'cm_{idx}']
        cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True).clip(1)
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(label, fontsize=14, fontweight='bold', pad=30)
        n = len(class_names)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=14)
        ax.set_yticklabels(class_names, fontsize=14)
        ax.set_xlabel("Predicted", fontsize=16)
        ax.set_ylabel("True", fontsize=16)

        for i in range(n):
            for j in range(n):
                val = cm_norm[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        fontsize=12, color=color)

    cb = fig.colorbar(im, cax=cbar_ax, label="Recall")
    cb.ax.yaxis.label.set_fontsize(20)
    cb.ax.tick_params(labelsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrices.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper figures from metrics.csv")
    parser.add_argument('--csv', type=str, default='outputs/figures/metrics.csv')
    parser.add_argument('--npz', type=str, default='outputs/figures/confusion_matrices.npz')
    parser.add_argument('--out', type=str, default='outputs/figures')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    models = read_metrics(args.csv)
    class_names = get_class_names(models)

    print(f"Loaded {len(models)} models from {args.csv}")
    print()

    # Print summary
    header = f"  {'Model':<20} {'OA':>8} {'mIoU':>8} {'mF1':>8} {'Kappa':>8}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for m in models:
        print(f"  {m['Model']:<20} {float(m['OA']):>8.4f} {float(m['mIoU']):>8.4f} "
              f"{float(m['mF1']):>8.4f} {float(m['Kappa']):>8.4f}")
    print()

    print("Generating figures...")
    fig_iou_comparison(models, class_names, args.out)
    fig_summary_table(models, class_names, args.out)

    if os.path.exists(args.npz):
        fig_confusion_matrices(args.npz, args.out)
    else:
        print(f"  Skipping confusion matrices ({args.npz} not found)")

    print(f"\nAll figures saved to: {args.out}/")


if __name__ == '__main__':
    main()
