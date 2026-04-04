"""
Unified training script for all model variants on Potsdam.

Supports:
  - arch: "moe_fusion"    (SkySense++ + optional Panopticon spatial + optional nDSM)
  - arch: "unet_baseline" (UNet-EfficientNet-B4, 5-ch RGBIR+nDSM input)

Ablation flags can be overridden from CLI:
  --no-panopticon   disable Panopticon spatial fusion
  --no-ndsm         disable nDSM fusion

Usage:
  docker compose run --rm moefusion pixi run python scripts/train.py --config configs/potsdam_moe.yaml
  docker compose run --rm moefusion pixi run python scripts/train.py --config configs/potsdam_moe.yaml --no-panopticon --no-ndsm
  docker compose run --rm moefusion pixi run python scripts/train.py --config configs/potsdam_baseline.yaml
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import yaml
import numpy as np


def compute_miou(pred, target, num_classes, ignore_index=0):
    ious = []
    for c in range(1, num_classes + 1):
        pred_c = pred == c
        target_c = target == c
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0, ious


def build_model(model_cfg, device):
    """Build model based on arch field in config.

    Returns (model, variant_name, forward_fn, set_eval_backbones).
    forward_fn(model, ms, ndsm, chn_ids, rgb_indices) -> dict with 'logits'
    """
    arch = model_cfg.get("arch", "moe_fusion")

    if arch == "unet_baseline":
        from core.models.baseline_unet import BaselineUNet
        model = BaselineUNet(
            num_classes=model_cfg["num_classes"],
            in_channels=model_cfg.get("in_channels", 5),
            encoder_name=model_cfg.get("encoder_name", "efficientnet-b4"),
            encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        ).to(device)
        variant_name = f"unet_{model_cfg.get('encoder_name', 'effb4')}"

        def forward_fn(model, ms, ndsm, chn_ids, rgb_indices):
            return model(torch.cat([ms, ndsm], dim=1))

        return model, variant_name, forward_fn, lambda m: None

    elif arch == "moe_fusion":
        from core.models.moe_segmentor import MoESegmentor
        use_pan = model_cfg.get("use_panopticon_spatial", True)
        use_ndsm = model_cfg.get("use_ndsm", True)
        model = MoESegmentor(
            num_classes=model_cfg["num_classes"],
            skysense_weights=model_cfg.get("skysense_weights"),
            img_size=model_cfg.get("img_size", 256),
            panopticon_checkpoint=model_cfg.get("panopticon_checkpoint"),
            fpn_channels=model_cfg.get("fpn_channels", 256),
            dropout=model_cfg.get("dropout", 0.1),
            use_panopticon_spatial=use_pan,
            use_ndsm=use_ndsm,
        ).to(device)

        parts = ["skysense"]
        if use_pan:
            parts.append("pan_spatial")
        if use_ndsm:
            parts.append("ndsm")
        variant_name = "+".join(parts)

        def forward_fn(model, ms, ndsm, chn_ids, rgb_indices):
            return model(ms, chn_ids,
                         x_ndsm=ndsm if use_ndsm else None,
                         rgb_indices=rgb_indices)

        def set_eval_backbones(model):
            model.skysense.eval()
            if use_pan:
                model.panopticon.eval()

        return model, variant_name, forward_fn, set_eval_backbones

    else:
        raise ValueError(f"Unknown arch: {arch}")


def run_epoch(model, loader, dm, criterion, forward_fn, chn_ids_base,
              device, ignore_index, optimizer=None, scheduler=None,
              params=None, log=None, epoch_str="", log_interval=300):
    """Shared train/val epoch logic. Pass optimizer=None for validation."""
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    epoch_losses = {'total': 0.0, 'ce': 0.0, 'lovasz': 0.0}
    correct = 0
    total_px = 0
    all_preds = [] if not training else None
    all_targets = [] if not training else None
    steps = len(loader)

    ctx = torch.no_grad() if not training else torch.enable_grad()
    with ctx:
        for step, (ms, ndsm, gt) in enumerate(loader):
            ms, ndsm, gt = ms.to(device), ndsm.to(device), gt.to(device)
            ms, ndsm = dm.normalize(ms, ndsm, device=device)
            if training:
                ms, ndsm, gt = dm.augment(ms, ndsm, gt)

            B = ms.shape[0]
            chn_ids = chn_ids_base[:B] if chn_ids_base is not None else None

            out = forward_fn(model, ms, ndsm, chn_ids, None)
            losses = criterion(out['logits'], gt)

            if training:
                optimizer.zero_grad()
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                scheduler.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()

            pred = out['logits'].argmax(dim=1)
            valid = gt != ignore_index
            correct += (pred[valid] == gt[valid]).sum().item()
            total_px += valid.sum().item()

            if not training:
                all_preds.append(pred.cpu())
                all_targets.append(gt.cpu())

            if training and log and (step + 1) % log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                log.info(f"  {epoch_str}[{step+1}/{steps}] "
                         f"loss={losses['total'].item():.4f} lr={lr:.6f}")

    acc = correct / max(1, total_px)
    for k in epoch_losses:
        epoch_losses[k] /= max(1, len(loader))

    if not training:
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        return epoch_losses, acc, all_preds, all_targets
    return epoch_losses, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/potsdam_moe.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no-panopticon', action='store_true',
                        help='Disable Panopticon spatial fusion')
    parser.add_argument('--no-ndsm', action='store_true',
                        help='Disable nDSM fusion')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides for ablation
    if args.no_panopticon:
        cfg["model"]["use_panopticon_spatial"] = False
    if args.no_ndsm:
        cfg["model"]["use_ndsm"] = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # -- Dataset -------------------------------------------------------------
    from core.datasets.potsdam import PotsdamDataModule

    dm = PotsdamDataModule(cfg)
    dm.setup()

    train_loader = dm.train_loader()
    val_loader = dm.val_loader()

    wavelengths = cfg["dataset"]["bands"]["wavelengths_nm"]
    rgb_indices = tuple(cfg["dataset"]["bands"]["rgb_indices"])
    ignore_index = cfg["dataset"].get("ignore_index", 0)

    # -- Model ---------------------------------------------------------------
    model_cfg = cfg["model"]
    model, variant_name, forward_fn, set_eval_backbones = build_model(
        model_cfg, device)

    # Pre-allocate chn_ids (max batch size) — slice per step
    use_pan = model_cfg.get("use_panopticon_spatial", False) and \
              model_cfg.get("arch", "moe_fusion") == "moe_fusion"
    max_bs = cfg["dataset"]["batch_size"]
    chn_ids_base = torch.tensor(
        [wavelengths] * max_bs, dtype=torch.float32, device=device
    ) if use_pan else None

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    # -- Loss ----------------------------------------------------------------
    from core.losses.losses import SegmentationLoss

    loss_cfg = cfg["training"]["loss"]
    criterion = SegmentationLoss(
        num_classes=model_cfg["num_classes"],
        ignore_index=ignore_index,
        ce_weight=loss_cfg.get("ce_weight", 1.0),
        lovasz_weight=loss_cfg.get("lovasz_weight", 1.0),
        label_smoothing=loss_cfg.get("label_smoothing", 0.1),
    ).to(device)

    # -- Optimizer + Scheduler -----------------------------------------------
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

    # -- Resume --------------------------------------------------------------
    start_epoch = 0
    best_miou = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_miou = ckpt.get('best_miou', 0.0)
        print(f"Resumed from epoch {start_epoch}, best mIoU={best_miou:.4f}")

    # -- Output dir + logging ------------------------------------------------
    out_dir = Path(f"outputs/{variant_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    fh = logging.FileHandler(out_dir / "train.log")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    val_interval = int(train_cfg.get("val_interval", 5))
    log_interval = int(train_cfg.get("log_interval", 50))

    log.info(f"Config: {args.config}")
    log.info(f"Variant: {variant_name}  |  arch: {model_cfg.get('arch', 'moe_fusion')}")
    log.info(f"Parameters -- Total: {total/1e6:.1f}M | Trainable: {trainable/1e6:.1f}M")
    log.info(f"Train: {len(dm.train_dataset)} | Val: {len(dm.val_dataset)} | "
             f"Test: {len(dm.test_dataset)}")
    log.info(f"Epochs: {max_epochs} | Val interval: {val_interval} | "
             f"LR: {train_cfg['lr']} | Batch: {cfg['dataset']['batch_size']}")

    # -- Training loop -------------------------------------------------------
    for epoch in range(start_epoch, max_epochs):
        set_eval_backbones(model)

        epoch_str = f"[{epoch+1}/{max_epochs}]"
        t0 = time.time()

        train_losses, train_acc = run_epoch(
            model, train_loader, dm, criterion, forward_fn, chn_ids_base,
            device, ignore_index, optimizer=optimizer, scheduler=scheduler,
            params=params, log=log, epoch_str=epoch_str,
            log_interval=log_interval)

        elapsed = time.time() - t0
        log.info(f"Epoch {epoch+1}/{max_epochs} ({elapsed:.0f}s) -- "
                 f"loss={train_losses['total']:.4f} OA={train_acc:.4f}")

        # -- Validation ------------------------------------------------------
        if (epoch + 1) % val_interval != 0 and epoch != max_epochs - 1:
            ckpt = {
                'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_miou': best_miou, 'config': cfg,
            }
            torch.save(ckpt, out_dir / 'last.pt')
            continue

        val_losses, val_acc, all_preds, all_targets = run_epoch(
            model, val_loader, dm, criterion, forward_fn, chn_ids_base,
            device, ignore_index)

        miou, per_class = compute_miou(
            all_preds, all_targets, model_cfg["num_classes"], ignore_index)

        log.info(f"  Val -- loss={val_losses['total']:.4f} "
                 f"OA={val_acc:.4f} mIoU={miou:.4f}")

        class_names = cfg["dataset"].get("class_names",
                                          [f"C{i}" for i in range(1, 7)])
        for name, iou in zip(class_names, per_class):
            log.info(f"    {name}: {iou:.4f}")

        # -- Checkpoint ------------------------------------------------------
        is_best = miou > best_miou
        if is_best:
            best_miou = miou

        ckpt = {
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_miou': best_miou, 'miou': miou, 'config': cfg,
        }
        torch.save(ckpt, out_dir / 'last.pt')
        if is_best:
            torch.save(ckpt, out_dir / 'best.pt')
            log.info(f"  ** New best mIoU: {best_miou:.4f} **")

        log.info("")

    log.info(f"Training complete. Best mIoU: {best_miou:.4f}")


if __name__ == '__main__':
    main()
