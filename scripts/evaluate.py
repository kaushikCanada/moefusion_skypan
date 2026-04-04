"""
Evaluate a trained model on the Potsdam test split.

Works for both MoE and baseline models.

Usage:
  # MoE
  python scripts/evaluate.py --config configs/potsdam_moe.yaml --checkpoint outputs/moe_fusion/best.pt --model moe

  # Baseline
  python scripts/evaluate.py --config configs/potsdam_baseline.yaml --checkpoint outputs/baseline/best.pt --model baseline
"""

import argparse
import yaml
import numpy as np
import torch


def compute_metrics(pred, target, num_classes, ignore_index=0):
    """Compute OA, mIoU, mF1, Cohen's Kappa, per-class IoU/F1."""
    valid = target != ignore_index
    p = pred[valid]
    t = target[valid]
    total = t.numel()

    oa = (p == t).float().mean().item()

    ious = []
    f1s = []
    for c in range(1, num_classes + 1):
        pred_c = p == c
        target_c = t == c
        tp = (pred_c & target_c).sum().item()
        fp = (pred_c & ~target_c).sum().item()
        fn = (~pred_c & target_c).sum().item()

        union = tp + fp + fn
        iou = tp / union if union > 0 else float('nan')
        ious.append(iou)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) \
            if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    valid_ious = [x for x in ious if not np.isnan(x)]
    miou = np.mean(valid_ious) if valid_ious else 0.0
    mf1 = np.mean(f1s) if f1s else 0.0

    # Cohen's Kappa
    pe = 0.0
    for c in range(num_classes + 1):
        if c == ignore_index:
            continue
        pe += ((p == c).sum().item() / total) * ((t == c).sum().item() / total)
    kappa = (oa - pe) / (1.0 - pe) if (1.0 - pe) > 0 else 0.0

    return oa, miou, mf1, kappa, ious, f1s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model', type=str, required=True,
                        choices=['moe', 'baseline'])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ignore_index = cfg["dataset"].get("ignore_index", 0)
    model_cfg = cfg["model"]

    # ── Dataset ─────────────────────────────────────────────────────────
    from core.datasets.potsdam import PotsdamDataModule

    dm = PotsdamDataModule(cfg)
    dm.setup()
    test_loader = dm.test_loader()
    print(f"Test set: {len(dm.test_dataset)} patches")

    # ── Model ───────────────────────────────────────────────────────────
    if args.model == 'moe':
        from core.models.moe_segmentor import MoESegmentor
        model = MoESegmentor(
            num_classes=model_cfg["num_classes"],
            skysense_weights=model_cfg.get("skysense_weights"),
            img_size=model_cfg.get("img_size", 256),
            panopticon_checkpoint=model_cfg.get("panopticon_checkpoint"),
            fpn_channels=model_cfg.get("fpn_channels", 256),
            dropout=0.0,
        ).to(device)
    else:
        from core.models.baseline_unet import BaselineUNet
        model = BaselineUNet(
            num_classes=model_cfg["num_classes"],
            in_channels=model_cfg.get("in_channels", 5),
            encoder_name=model_cfg.get("encoder_name", "efficientnet-b4"),
        ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    model.eval()

    # ── Config for MoE forward ──────────────────────────────────────────
    wavelengths = cfg["dataset"]["bands"]["wavelengths_nm"]
    rgb_indices = tuple(cfg["dataset"]["bands"]["rgb_indices"])

    # ── Inference ───────────────────────────────────────────────────────
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (ms, ndsm, gt) in enumerate(test_loader):
            ms, ndsm, gt = ms.to(device), ndsm.to(device), gt.to(device)
            ms, ndsm = dm.normalize(ms, ndsm, device=device)

            if args.model == 'moe':
                B = ms.shape[0]
                chn_ids = torch.tensor(
                    [wavelengths] * B, dtype=torch.float32, device=device)
                out = model(ms, chn_ids, ndsm, rgb_indices=rgb_indices)
                logits = out['logits']
            else:
                x = torch.cat([ms, ndsm], dim=1)
                logits = model(x)

            all_preds.append(logits.argmax(dim=1).cpu())
            all_targets.append(gt.cpu())

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(test_loader)} batches...")

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # ── Metrics ─────────────────────────────────────────────────────────
    oa, miou, mf1, kappa, ious, f1s = compute_metrics(
        all_preds, all_targets, model_cfg["num_classes"], ignore_index)

    class_names = cfg["dataset"].get("class_names",
                                     [f"C{i}" for i in range(1, 7)])

    print(f"\n{'='*60}")
    print(f"  Model:  {args.model}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Test patches: {len(dm.test_dataset)}")
    print(f"{'='*60}")
    print(f"  OA:    {oa:.4f}")
    print(f"  mIoU:  {miou:.4f}")
    print(f"  mF1:   {mf1:.4f}")
    print(f"  Kappa: {kappa:.4f}")
    print(f"{'='*60}")
    print(f"  {'Class':<25} {'IoU':>8} {'F1':>8}")
    print(f"  {'-'*41}")
    for name, iou, f1 in zip(class_names, ious, f1s):
        iou_str = f"{iou:.4f}" if not np.isnan(iou) else "  N/A "
        print(f"  {name:<25} {iou_str:>8} {f1:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
