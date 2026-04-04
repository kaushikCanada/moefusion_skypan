"""Smoke test: MoE Segmentor end-to-end forward pass."""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skysense-weights', type=str, default=None)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--num-classes', type=int, default=6)
    args = parser.parse_args()

    from core.models.moe_segmentor import MoESegmentor

    # ── Dataset config (would come from YAML in real training) ──────────
    dataset_cfg = {
        'name': 'potsdam',
        'bands': ['R', 'G', 'B', 'IR'],
        'wavelengths_nm': [660.0, 550.0, 470.0, 840.0],
        'rgb_indices': (0, 1, 2),
        'num_classes': args.num_classes,
    }

    print("Building MoESegmentor...")
    print(f"  img_size={args.img_size}, num_classes={dataset_cfg['num_classes']}")
    print(f"  bands={dataset_cfg['bands']}, rgb_indices={dataset_cfg['rgb_indices']}")

    model = MoESegmentor(
        num_classes=dataset_cfg['num_classes'],
        skysense_weights=args.skysense_weights,
        img_size=args.img_size,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"  Total: {total / 1e6:.1f}M | Trainable: {trainable / 1e6:.1f}M "
          f"| Frozen: {frozen / 1e6:.1f}M")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    B = 2
    C = len(dataset_cfg['bands'])
    x_msi = torch.randn(B, C, args.img_size, args.img_size, device=device)
    chn_ids = torch.tensor(
        [dataset_cfg['wavelengths_nm']] * B,
        dtype=torch.float32, device=device)
    x_ndsm = torch.randn(B, 1, args.img_size, args.img_size, device=device)

    # ── Test WITH nDSM ──────────────────────────────────────────────────
    print(f"\nForward pass WITH nDSM:")
    print(f"  x_msi:  {tuple(x_msi.shape)} (bands: {dataset_cfg['bands']})")
    print(f"  chn_ids:{tuple(chn_ids.shape)} (nm: {dataset_cfg['wavelengths_nm']})")
    print(f"  x_ndsm: {tuple(x_ndsm.shape)}")

    out = model(x_msi, chn_ids, x_ndsm,
                rgb_indices=dataset_cfg['rgb_indices'])
    print(f"  logits:       {tuple(out['logits'].shape)}")
    print(f"  gate_weights: {[f'{w:.3f}' for w in out['gate_weights'][0].detach().cpu().tolist()]}")

    assert out['logits'].shape == (B, args.num_classes, args.img_size,
                                   args.img_size)

    # ── Test WITHOUT nDSM ───────────────────────────────────────────────
    print(f"\nForward pass WITHOUT nDSM:")
    out2 = model(x_msi, chn_ids, x_ndsm=None,
                 rgb_indices=dataset_cfg['rgb_indices'])
    print(f"  logits:       {tuple(out2['logits'].shape)}")
    print(f"  gate_weights: {[f'{w:.3f}' for w in out2['gate_weights'][0].detach().cpu().tolist()]}")

    assert out2['logits'].shape == (B, args.num_classes, args.img_size,
                                    args.img_size)

    # ── Verify trainable modules ────────────────────────────────────────
    trainable_modules = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            mod = name.split('.')[0]
            if mod not in trainable_modules:
                trainable_modules.append(mod)
    print(f"\nTrainable modules: {trainable_modules}")

    # ── Verify backbones are frozen ─────────────────────────────────────
    for name, p in model.skysense.named_parameters():
        assert not p.requires_grad, f"SKY++ param not frozen: {name}"
    for name, p in model.panopticon.named_parameters():
        assert not p.requires_grad, f"Panopticon param not frozen: {name}"
    print("Backbone freeze verified.")

    # ── Gate weights sanity check ───────────────────────────────────────
    gw = out['gate_weights'][0].detach().cpu()
    assert abs(gw.mean().item() - 1.0) < 0.3, \
        f"Gate weights mean unexpected: {gw}"
    assert (gw > 0).all(), "Gate weights should be positive"
    print(f"Gate weights sanity OK (mean={gw.mean().item():.3f}).")

    # ── Verify gradient flow (with nDSM) ─────────────────────────────────
    print("\nChecking gradient flow (with nDSM)...")
    model.zero_grad()
    loss = out['logits'].mean()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"
    print("  Gradients OK for all trainable parameters.")

    # ── Verify gradient flow (without nDSM) ───────────────────────────
    print("Checking gradient flow (without nDSM)...")
    model.zero_grad()
    loss2 = out2['logits'].mean()
    loss2.backward()
    for name, p in model.named_parameters():
        if p.requires_grad and 'ndsm' not in name:
            assert p.grad is not None, f"No gradient (no-nDSM path) for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"
    print("  Gradients OK for no-nDSM path.")

    print("\nSmoke test passed!")


if __name__ == '__main__':
    main()
