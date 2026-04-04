"""Smoke test: instantiate SkySense++ HR backbone and optionally load weights."""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to skysense_model_backbone_hr.pth')
    parser.add_argument('--img-size', type=int, default=224)
    args = parser.parse_args()

    from core.models.skysense_backbone import build_skysense_hr_backbone

    print(f"Building SkySense++ HR backbone (SwinV2-Huge)...")
    print(f"  in_channels=3 (RGB), img_size={args.img_size}")

    model = build_skysense_hr_backbone(
        pretrained_path=args.weights,
        img_size=args.img_size,
        frozen=True,
    )

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total / 1e6:.1f}M")

    # Test forward pass
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    x = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    print(f"\nForward pass with input shape: {tuple(x.shape)}")

    with torch.no_grad():
        outs = model(x)  # no annotation image → skip MSL path

    print(f"\nOutput feature maps:")
    for i, feat in enumerate(outs):
        print(f"  Stage {i}: {tuple(feat.shape)}")

    # Expected for Huge @ 224×224:
    # Stage 0: (1,  352, 56, 56)
    # Stage 1: (1,  704, 28, 28)
    # Stage 2: (1, 1408, 14, 14)
    # Stage 3: (1, 2816,  7,  7)

    print("\nSmoke test passed!")


if __name__ == '__main__':
    main()
