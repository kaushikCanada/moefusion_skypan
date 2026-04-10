"""
Evaluate multiple models on test set and print metrics table.

Usage:
  python scripts/evaluate.py --eval-config configs/evaluate.yaml --out outputs/figures
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train import build_model

CLASS_NAMES = ["Impervious", "Building", "Low veg", "Tree", "Car"]


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

    ious, f1s = [], []
    for c in range(1, len(CLASS_NAMES) + 1):
        pc, tc = p == c, t == c
        tp = (pc & tc).sum().item()
        fp = (pc & ~tc).sum().item()
        fn = (~pc & tc).sum().item()
        union = tp + fp + fn
        ious.append(tp / union if union > 0 else 0.0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)

    miou = np.mean([x for x in ious if x > 0])
    mf1 = np.mean(f1s)

    pe = sum(((p == c).sum().item() / total) * ((t == c).sum().item() / total)
             for c in range(1, len(CLASS_NAMES) + 1))
    kappa = (oa - pe) / (1.0 - pe) if (1.0 - pe) > 0 else 0.0

    return {
        'oa': oa, 'miou': miou, 'mf1': mf1, 'kappa': kappa,
        'ious': ious, 'f1s': f1s,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on test set and print metrics table.")
    parser.add_argument('--eval-config', type=str, required=True,
                        help='Path to evaluate.yaml')
    parser.add_argument('--out', type=str, default='outputs/figures')
    args = parser.parse_args()

    with open(args.eval_config) as f:
        eval_cfg = yaml.safe_load(f)

    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset from dataset_config
    with open(eval_cfg["dataset_config"]) as f:
        data_cfg = yaml.safe_load(f)

    ignore_index = data_cfg["dataset"].get("ignore_index", 0)
    wavelengths = data_cfg["dataset"]["bands"]["wavelengths_nm"]

    from core.datasets.potsdam import PotsdamDataModule
    dm = PotsdamDataModule(data_cfg)
    dm.setup()
    test_loader = dm.test_loader()

    # Load all models
    labels = []
    models = {}
    forward_fns = {}
    for entry in eval_cfg["models"]:
        label = entry["label"]
        labels.append(label)
        print(f"Loading {label}...")
        with open(entry["config"]) as f:
            cfg = yaml.safe_load(f)
        model, variant, forward_fn = load_model_from_config(
            cfg, entry["checkpoint"], device)
        models[label] = model
        forward_fns[label] = forward_fn

    max_bs = data_cfg["dataset"]["batch_size"]
    chn_ids_base = torch.tensor(
        [wavelengths] * max_bs, dtype=torch.float32, device=device)

    # Run inference
    print(f"\nRunning inference on test set ({len(test_loader)} batches)...")
    all_preds = {label: [] for label in labels}
    all_targets = []

    with torch.no_grad():
        for i, (ms, ndsm, ndvi, gt) in enumerate(test_loader):
            ms, ndsm, ndvi, gt = ms.to(device), ndsm.to(device), ndvi.to(device), gt.to(device)
            ms, ndsm = dm.normalize(ms, ndsm, device=device)

            B = ms.shape[0]
            chn_ids = chn_ids_base[:B]

            for label in labels:
                out = forward_fns[label](models[label], ms, ndsm, ndvi, chn_ids, None)
                pred = out['logits'].argmax(dim=1)
                all_preds[label].append(pred.cpu())

            all_targets.append(gt.cpu())

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(test_loader)} batches...")

    for label in labels:
        all_preds[label] = torch.cat(all_preds[label])
    all_targets = torch.cat(all_targets)

    # Compute metrics
    print("\nComputing metrics...")
    all_metrics = {}
    for label in labels:
        all_metrics[label] = compute_full_metrics(
            all_preds[label], all_targets, ignore_index)

    # Print table
    header = f"  {'Metric':<12}"
    for label in labels:
        header += f" {label:>16}"
    print(f"\n{'=' * len(header)}")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for metric_name, key in [('OA', 'oa'), ('mIoU', 'miou'),
                              ('mF1', 'mf1'), ('Kappa', 'kappa')]:
        row = f"  {metric_name:<12}"
        for label in labels:
            row += f" {all_metrics[label][key]:>16.4f}"
        print(row)

    print(f"  {'-' * (len(header) - 2)}")
    print(f"  {'Class':<12}", end="")
    for label in labels:
        print(f" {'IoU':>8}{'F1':>8}", end="")
    print()

    for i, name in enumerate(CLASS_NAMES):
        row = f"  {name:<12}"
        for label in labels:
            row += f" {all_metrics[label]['ious'][i]:>8.4f}"
            row += f"{all_metrics[label]['f1s'][i]:>8.4f}"
        print(row)

    print(f"{'=' * len(header)}")

    # Compute and save confusion matrices
    print("Computing confusion matrices...")
    cms = {}
    n = len(CLASS_NAMES)
    for label in labels:
        cm = np.zeros((n, n), dtype=np.int64)
        p = all_preds[label][all_targets != ignore_index].numpy()
        t = all_targets[all_targets != ignore_index].numpy()
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                cm[i-1, j-1] = ((t == i) & (p == j)).sum()
        cms[label] = cm

    npz_path = os.path.join(args.out, "confusion_matrices.npz")
    np.savez(npz_path, labels=np.array(labels), class_names=np.array(CLASS_NAMES),
             **{f"cm_{i}": cms[label] for i, label in enumerate(labels)})
    print(f"Confusion matrices saved to: {npz_path}")

    # Save CSV
    csv_path = os.path.join(args.out, "metrics.csv")
    with open(csv_path, 'w') as f:
        f.write("Model,OA,mIoU,mF1,Kappa," +
                ",".join(f"IoU_{c}" for c in CLASS_NAMES) + "," +
                ",".join(f"F1_{c}" for c in CLASS_NAMES) + "\n")
        for label in labels:
            m = all_metrics[label]
            f.write(f"{label},{m['oa']:.4f},{m['miou']:.4f},"
                    f"{m['mf1']:.4f},{m['kappa']:.4f},")
            f.write(",".join(f"{x:.4f}" for x in m['ious']))
            f.write(",")
            f.write(",".join(f"{x:.4f}" for x in m['f1s']))
            f.write("\n")
    print(f"\nMetrics saved to: {csv_path}")


if __name__ == '__main__':
    main()
