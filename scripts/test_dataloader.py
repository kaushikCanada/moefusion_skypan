"""Verify Potsdam dataloader: setup, iterate, print N random samples."""

import argparse
import random
import yaml
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/potsdam.yaml')
    parser.add_argument('--n', type=int, default=5, help='Number of samples to print')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from core.datasets.potsdam import PotsdamDataModule

    print("Setting up PotsdamDataModule...")
    dm = PotsdamDataModule(cfg)
    dm.setup()

    print(f"\nWavelengths (nm): {dm.wavelengths_nm}")
    print(f"RGB indices:      {dm.rgb_indices}")
    print(f"Num classes:      {dm.num_classes}")

    # Print N random samples from train
    train_ds = dm.train_dataset
    indices = random.sample(range(len(train_ds)), min(args.n, len(train_ds)))

    print(f"\n{'='*60}")
    print(f"Printing {len(indices)} random TRAIN samples:")
    print(f"{'='*60}")
    for i, idx in enumerate(indices):
        sample = train_ds[idx]
        ms = sample["ms"]
        ndsm = sample["ndsm"]
        ndvi = sample["ndvi"]
        gt = sample["gt"]
        tile = sample["tile_id"]

        # Class distribution
        classes, counts = gt.unique(return_counts=True)
        total = gt.numel()
        dist = {c.item(): f"{count.item()/total*100:.1f}%" for c, count in zip(classes, counts)}

        print(f"\n  Sample {i+1} (index={idx}, tile={tile}):")
        print(f"    ms:   shape={tuple(ms.shape)}, dtype={ms.dtype}, "
              f"min={ms.min():.1f}, max={ms.max():.1f}, mean={ms.mean():.1f}")
        print(f"    ndsm: shape={tuple(ndsm.shape)}, dtype={ndsm.dtype}, "
              f"min={ndsm.min():.2f}, max={ndsm.max():.2f}, mean={ndsm.mean():.2f}")
        print(f"    ndvi: shape={tuple(ndvi.shape)}, dtype={ndvi.dtype}, "
              f"min={ndvi.min():.2f}, max={ndvi.max():.2f}, mean={ndvi.mean():.2f}")
        print(f"    gt:   shape={tuple(gt.shape)}, dtype={gt.dtype}, "
              f"classes={dist}")

    # Test a single batch from train loader
    print(f"\n{'='*60}")
    print("Testing train dataloader batch...")
    print(f"{'='*60}")
    loader = dm.train_loader()
    ms, ndsm, ndvi, gt = next(iter(loader))
    print(f"  Batch ms:   {tuple(ms.shape)}")
    print(f"  Batch ndsm: {tuple(ndsm.shape)}")
    print(f"  Batch ndvi: {tuple(ndvi.shape)}")
    print(f"  Batch gt:   {tuple(gt.shape)}")

    # Test normalization
    ms_n, ndsm_n = dm.normalize(ms, ndsm)
    print(f"\n  After normalize:")
    print(f"    ms:   min={ms_n.min():.2f}, max={ms_n.max():.2f}, mean={ms_n.mean():.2f}")
    print(f"    ndsm: min={ndsm_n.min():.2f}, max={ndsm_n.max():.2f}, mean={ndsm_n.mean():.2f}")
    print(f"    ndvi: min={ndvi.min():.2f}, max={ndvi.max():.2f}, mean={ndvi.mean():.2f} (no normalization)")

    # Test augmentation
    ms_a, ndsm_a, ndvi_a, gt_a = dm.augment(ms_n, ndsm_n, ndvi, gt)
    print(f"\n  After augment:")
    print(f"    ms:   {tuple(ms_a.shape)}")
    print(f"    ndsm: {tuple(ndsm_a.shape)}")
    print(f"    ndvi: {tuple(ndvi_a.shape)}")
    print(f"    gt:   {tuple(gt_a.shape)}")

    # Test val loader
    print(f"\n{'='*60}")
    print("Testing val dataloader...")
    print(f"{'='*60}")
    val_loader = dm.val_loader()
    ms_v, ndsm_v, ndvi_v, gt_v = next(iter(val_loader))
    print(f"  Val batch: ms={tuple(ms_v.shape)}, ndsm={tuple(ndsm_v.shape)}, ndvi={tuple(ndvi_v.shape)}, gt={tuple(gt_v.shape)}")

    print("\nDataloader test passed!")


if __name__ == '__main__':
    main()
