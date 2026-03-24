from __future__ import annotations

import argparse

from models import build_model
from utils.flops import compute_model_complexity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute model FLOPs / MACs")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--use-pretrained", action="store_true")
    parser.add_argument("--freeze-early-layers", action="store_true")
    parser.add_argument("--image-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = build_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        use_pretrained=args.use_pretrained,
        freeze_early_layers=args.freeze_early_layers,
    )

    macs, params = compute_model_complexity(
        model=model,
        input_shape=(3, args.image_size, args.image_size),
    )

    print(f"Model: {args.model_name}")
    print(f"Input size: 3x{args.image_size}x{args.image_size}")
    print(f"MACs: {macs}")
    print(f"Params: {params}")


if __name__ == "__main__":
    main()