"""Smoke test: end-to-end forward pass through frozen SkySense++ → FPN → UPerNet."""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to skysensepp_release_hr.pth')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--num-classes', type=int, default=6)
    args = parser.parse_args()

    from core.models.segmentor import SkySenseSegmentor

    print(f"Building SkySenseSegmentor...")
    print(f"  img_size={args.img_size}, num_classes={args.num_classes}")

    model = SkySenseSegmentor(
        num_classes=args.num_classes,
        pretrained_path=args.weights,
        img_size=args.img_size,
    )

    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"  Total: {total / 1e6:.1f}M | Trainable: {trainable / 1e6:.1f}M | Frozen: {frozen / 1e6:.1f}M")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    x = torch.randn(2, 3, args.img_size, args.img_size, device=device)
    print(f"\nForward pass with input: {tuple(x.shape)}")

    logits = model(x)
    print(f"Output logits: {tuple(logits.shape)}")

    assert logits.shape == (2, args.num_classes, args.img_size, args.img_size), \
        f"Expected (2, {args.num_classes}, {args.img_size}, {args.img_size}), got {tuple(logits.shape)}"

    print("\nSmoke test passed!")


if __name__ == '__main__':
    main()
