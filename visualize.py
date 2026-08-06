"""
visualize.py
------------
Standalone visualization entrypoint for Shockwave PINN.

This script loads a saved checkpoint and normaliser, performs PINN inference
on either synthetic or SU2 data, and saves the standard plots without
running training.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
from pathlib import Path

from main import load_raw_data, build_model, load_checkpoint, inference_and_plot
from data.loader import DataNormalizer


def build_parser():
    p = argparse.ArgumentParser(description="Shockwave PINN visualization only")
    p.add_argument("--mode", choices=["synthetic", "su2"],
                   default="synthetic",
                   help="Data source for inference and plotting.")
    p.add_argument("--checkpoint", required=True,
                   help="Saved model checkpoint to load.")
    p.add_argument("--normalizer", default=None,
                   help="Saved normalizer .npz file. Defaults to outputs/normalizer.npz.")
    p.add_argument("--data_dir", default=".",
                   help="Data directory for SU2 reference output.")
    p.add_argument("--output_dir", default="outputs",
                   help="Directory for saved plots.")
    p.add_argument("--device", default="auto",
                   help="cpu, cuda, or auto.")
    p.add_argument("--mach_inf", type=float, default=2.5)
    p.add_argument("--wedge_angle_deg", type=float, default=15.0)
    p.add_argument("--p_inf", type=float, default=101325.0)
    p.add_argument("--T_inf", type=float, default=300.0)
    p.add_argument("--rho_inf", type=float, default=1.225)
    p.add_argument("--n_points", type=int, default=8000)
    p.add_argument("--hidden_layers", type=int, default=8)
    p.add_argument("--hidden_width", type=int, default=128)
    return p


def parse_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def main():
    args = build_parser().parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = parse_device(args.device)
    print(f"\n{'='*60}")
    print(f"  Shockwave PINN  |  visualize only  |  mode={args.mode}  |  device={device}")
    print(f"{'='*60}\n")

    raw_data, beta_rad, theta_rad = load_raw_data(args)

    model = build_model(args)
    model = load_checkpoint(model, args.checkpoint, device)
    print(f"[Visualize] Loaded checkpoint: {args.checkpoint}")

    norm_path = Path(args.normalizer) if args.normalizer else outdir / "normalizer.npz"
    if not norm_path.exists():
        raise FileNotFoundError(
            f"Normaliser file not found: {norm_path}. Use --normalizer or run training first."
        )
    normalizer = DataNormalizer.load(str(norm_path))
    print(f"[Visualize] Loaded normalizer: {norm_path}")

    inference_and_plot(args, model, normalizer, raw_data, beta_rad, theta_rad, outdir, device)
    print(f"\n[Visualize] Done. Outputs in: {outdir}/")


if __name__ == "__main__":
    main()
