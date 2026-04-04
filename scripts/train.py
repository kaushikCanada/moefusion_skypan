"""
Training script for MoE Fusion Segmentor on Potsdam.

Usage:
  docker compose run --rm moefusion pixi run python scripts/train.py --config configs/potsdam.yaml
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
import numpy as np


def compute_miou(pred, target, num_classes, ignore_index=0):
    """Compute per-class IoU and mIoU."""
    ious = []
    for c in range(1, num_classes + 1):  # skip ignore_index=0
        pred_c = pred == c
        target_c = target == c
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0, ious


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/potsdam.yaml')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Dataset ─────────────────────────────────────────────────────────
    from core.datasets.potsdam import PotsdamDataModule

    dm = PotsdamDataModule(cfg)
    dm.setup()

    train_loader = dm.train_loader()
    val_loader = dm.val_loader()

    # Constant chn_ids tensor (same for all batches)
    wavelengths = cfg["dataset"]["bands"]["wavelengths_nm"]
    rgb_indices = tuple(cfg["dataset"]["bands"]["rgb_indices"])
    ignore_index = cfg["dataset"].get("ignore_index", 0)

    # ── Model ───────────────────────────────────────────────────────────
    from core.models.moe_segmentor import MoESegmentor

    model_cfg = cfg["model"]
    model = MoESegmentor(
        num_classes=model_cfg["num_classes"],
        skysense_weights=model_cfg.get("skysense_weights"),
        img_size=model_cfg.get("img_size", 256),
        panopticon_checkpoint=model_cfg.get("panopticon_checkpoint"),
        fpn_channels=model_cfg.get("fpn_channels", 256),
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    # ── Loss ────────────────────────────────────────────────────────────
    from core.losses.losses import MoEFusionLoss

    loss_cfg = cfg["training"]["loss"]
    criterion = MoEFusionLoss(
        num_classes=model_cfg["num_classes"],
        ignore_index=ignore_index,
        ce_weight=loss_cfg.get("ce_weight", 1.0),
        lovasz_weight=loss_cfg.get("lovasz_weight", 1.0),
        gate_entropy_weight=loss_cfg.get("gate_entropy_weight", 0.1),
        label_smoothing=loss_cfg.get("label_smoothing", 0.1),
    ).to(device)

    # ── Optimizer + Scheduler ───────────────────────────────────────────
    train_cfg = cfg["training"]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    max_epochs = int(train_cfg["max_epochs"])
    warmup_epochs = int(train_cfg.get("warmup_epochs", 2))
    steps_per_epoch = len(train_loader)
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume ──────────────────────────────────────────────────────────
    start_epoch = 0
    best_miou = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_miou = ckpt.get('best_miou', 0.0)
        print(f"Resumed from epoch {start_epoch}, best mIoU={best_miou:.4f}")  # before logger init

    # ── Output dir + logging ───────────────────────────────────────────
    out_dir = Path("outputs/moe_fusion")
    out_dir.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    # File
    fh = logging.FileHandler(out_dir / "train.log")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    val_interval = int(train_cfg.get("val_interval", 5))
    log_interval = int(train_cfg.get("log_interval", 50))

    log.info(f"Config: {args.config}")
    log.info(f"Parameters — Total: {total/1e6:.1f}M | Trainable: {trainable/1e6:.1f}M")
    log.info(f"Train: {len(dm.train_dataset)} | Val: {len(dm.val_dataset)} | "
             f"Test: {len(dm.test_dataset)}")
    log.info(f"Epochs: {max_epochs} | Val interval: {val_interval} | "
             f"LR: {train_cfg['lr']} | Batch: {cfg['dataset']['batch_size']}")

    # ── Training loop ───────────────────────────────────────────────────
    for epoch in range(start_epoch, max_epochs):
        model.train()
        # Keep backbones in eval mode
        model.skysense.eval()
        model.panopticon.eval()

        epoch_losses = {'total': 0, 'ce': 0, 'lovasz': 0, 'gate_entropy': 0}
        train_correct = 0
        train_total = 0
        t0 = time.time()

        for step, (ms, ndsm, gt) in enumerate(train_loader):
            ms, ndsm, gt = ms.to(device), ndsm.to(device), gt.to(device)

            # Normalize
            ms, ndsm = dm.normalize(ms, ndsm, device=device)

            # Augment
            ms, ndsm, gt = dm.augment(ms, ndsm, gt)

            # chn_ids: (B, C) — same wavelengths for every sample
            B = ms.shape[0]
            chn_ids = torch.tensor(
                [wavelengths] * B, dtype=torch.float32, device=device)

            # Forward
            out = model(ms, chn_ids, ndsm, rgb_indices=rgb_indices)

            # Loss
            losses = criterion(out['logits'], gt, out['gate_weights'])

            # Backward
            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()

            # Track training accuracy (exclude ignore pixels)
            with torch.no_grad():
                pred = out['logits'].argmax(dim=1)
                valid = gt != ignore_index
                train_correct += (pred[valid] == gt[valid]).sum().item()
                train_total += valid.sum().item()

            if (step + 1) % log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                gw = out['gate_weights'][0].detach().cpu()
                log.info(f"  [{epoch+1}/{max_epochs}][{step+1}/{steps_per_epoch}] "
                         f"loss={losses['total'].item():.4f} lr={lr:.6f} "
                         f"gw=[{', '.join(f'{w:.2f}' for w in gw.tolist())}]")

        # Epoch summary
        n = len(train_loader)
        elapsed = time.time() - t0
        train_acc = train_correct / max(1, train_total)
        log.info(f"Epoch {epoch+1}/{max_epochs} ({elapsed:.0f}s) — "
                 f"loss={epoch_losses['total']/n:.4f} OA={train_acc:.4f}")

        # ── Validation ──────────────────────────────────────────────────
        if (epoch + 1) % val_interval != 0 and epoch != max_epochs - 1:
            # Save last checkpoint even without val
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_miou': best_miou,
            }
            torch.save(ckpt, out_dir / 'last.pt')
            continue

        model.eval()
        val_losses = {'total': 0, 'ce': 0, 'lovasz': 0, 'gate_entropy': 0}
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for ms, ndsm, gt in val_loader:
                ms, ndsm, gt = ms.to(device), ndsm.to(device), gt.to(device)
                ms, ndsm = dm.normalize(ms, ndsm, device=device)

                B = ms.shape[0]
                chn_ids = torch.tensor(
                    [wavelengths] * B, dtype=torch.float32, device=device)

                out = model(ms, chn_ids, ndsm, rgb_indices=rgb_indices)
                losses = criterion(out['logits'], gt, out['gate_weights'])

                for k in val_losses:
                    val_losses[k] += losses[k].item()

                pred = out['logits'].argmax(dim=1)  # (B, H, W)
                all_preds.append(pred.cpu())
                all_targets.append(gt.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        miou, per_class = compute_miou(
            all_preds, all_targets, model_cfg["num_classes"], ignore_index)

        # Overall accuracy (excluding ignore)
        valid = all_targets != ignore_index
        val_acc = (all_preds[valid] == all_targets[valid]).float().mean().item()

        nv = len(val_loader)
        log.info(f"  Val — loss={val_losses['total']/nv:.4f} "
                 f"OA={val_acc:.4f} mIoU={miou:.4f}")

        class_names = cfg["dataset"].get("class_names", [f"C{i}" for i in range(1, 7)])
        for i, (name, iou) in enumerate(zip(class_names, per_class)):
            log.info(f"    {name}: {iou:.4f}")

        # ── Checkpoint ──────────────────────────────────────────────────
        is_best = miou > best_miou
        if is_best:
            best_miou = miou

        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_miou': best_miou,
            'miou': miou,
        }
        torch.save(ckpt, out_dir / 'last.pt')
        if is_best:
            torch.save(ckpt, out_dir / 'best.pt')
            log.info(f"  ** New best mIoU: {best_miou:.4f} **")

        log.info("")

    log.info(f"Training complete. Best mIoU: {best_miou:.4f}")


if __name__ == '__main__':
    main()
