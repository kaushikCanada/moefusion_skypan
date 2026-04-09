"""
Plot label efficiency and convergence charts from results.md data.

Usage:
  python scripts/plot_results.py --out outputs/figures
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_label_efficiency(out_dir):
    fractions = [5, 10, 25, 50, 100]

    data = {
        'UNet-EffB4':    [0.1995, 0.2978, 0.6934, 0.7320, 0.7531],
        'U-Panopticon':  [0.7024, 0.7396, 0.7632, 0.7867, 0.7948],
        'SegFormer-B2':  [0.4759, 0.5301, 0.6030, 0.6471, 0.6957],
        'SegFormer+Pan': [0.7342, 0.7603, 0.7769, 0.7924, 0.8017],
    }

    colors = {
        'UNet-EffB4':    '#FF9800',
        'U-Panopticon':  '#2196F3',
        'SegFormer-B2':  '#E91E63',
        'SegFormer+Pan': '#4CAF50',
    }
    markers = {
        'UNet-EffB4':    'o',
        'U-Panopticon':  's',
        'SegFormer-B2':  '^',
        'SegFormer+Pan': 'D',
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for model, mious in data.items():
        ax.plot(fractions, mious, marker=markers[model], color=colors[model],
                linewidth=2.5, markersize=10, label=model)
        # Value labels
        for x, y in zip(fractions, mious):
            ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=10)

    ax.set_xlabel('Training Data (%)', fontsize=16)
    ax.set_ylabel('mIoU', fontsize=16)
    ax.set_title('Label Efficiency', fontsize=20, fontweight='bold', pad=20)
    ax.set_xticks(fractions)
    ax.set_xticklabels([f'{f}%' for f in fractions], fontsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(fontsize=13, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_ylim(0.1, 0.85)

    plt.tight_layout()
    path = os.path.join(out_dir, "label_efficiency.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_convergence(out_dir):
    epochs = [10, 20, 30]

    data = {
        'UNet-EffB4':    [0.6858, 0.7265, 0.7531],
        'U-Panopticon':  [0.7772, 0.7823, 0.7948],
        'SegFormer-B2':  [0.6232, 0.6756, 0.6957],
        'SegFormer+Pan': [0.7950, 0.7966, 0.8017],
    }

    colors = {
        'UNet-EffB4':    '#FF9800',
        'U-Panopticon':  '#2196F3',
        'SegFormer-B2':  '#E91E63',
        'SegFormer+Pan': '#4CAF50',
    }
    markers = {
        'UNet-EffB4':    'o',
        'U-Panopticon':  's',
        'SegFormer-B2':  '^',
        'SegFormer+Pan': 'D',
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for model, mious in data.items():
        ax.plot(epochs, mious, marker=markers[model], color=colors[model],
                linewidth=2.5, markersize=10, label=model)
        for x, y in zip(epochs, mious):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=10)

    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('Val mIoU', fontsize=16)
    ax.set_title('Convergence', fontsize=20, fontweight='bold', pad=20)
    ax.set_xticks(epochs)
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(fontsize=13, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_ylim(0.55, 0.85)

    plt.tight_layout()
    path = os.path.join(out_dir, "convergence.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='outputs/figures')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Plotting results...")
    plot_label_efficiency(args.out)
    plot_convergence(args.out)
    print("Done.")


if __name__ == '__main__':
    main()
