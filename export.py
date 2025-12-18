#!/usr/bin/env python3
"""
Export trained model weights for FPGA without retraining.

Usage examples:

  python utils/export.py \
    --checkpoint checkpoints/final_model.pt \
    --export_dir ./export

  python utils/export.py \
    --checkpoint checkpoints/checkpoint_epoch_50.pt \
    --export_dir ./export

This script loads the generator from a training checkpoint and calls
utils.quantization.export_weights_fpga to emit binary weight/scale files.
"""

import argparse
import os
from pathlib import Path

import torch

from models.generator import MiniGenerator
from utils.quantization import QuantizationConfig, export_weights_fpga


def parse_args():
    parser = argparse.ArgumentParser(description="Export generator weights for FPGA")
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to training checkpoint (.pt)")
    parser.add_argument("--export_dir", type=str, default="./export",
                        help="Directory to write exported weight files")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", None],
                        help="Device to load checkpoint (default: auto)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Select device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Create model and load weights
    generator = MiniGenerator()
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Support both full training checkpoint and bare state_dict
    state_dict = checkpoint.get("generator_state_dict", checkpoint)
    generator.load_state_dict(state_dict)
    generator.eval()

    # Prepare export directory
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting weights from: {args.checkpoint}")
    print(f"Output directory:       {export_dir}")

    # Run export (quantization handles CPU moves for numpy conversion)
    quant_config = QuantizationConfig()
    export_weights_fpga(generator, str(export_dir / "generator"), quant_config)

    print("Export complete.")


if __name__ == "__main__":
    main()
