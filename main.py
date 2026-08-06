"""
main.py
-------
Shockwave PINN — end-to-end pipeline.

Usage
-----
# Synthetic data (no reference CFD needed):
    python main.py --mode synthetic

# SU2 CSV reference data:
    python main.py --mode su2 --data_dir path/to/su2/output.csv

# Inference only from saved checkpoint:
    python main.py --mode inference --checkpoint outputs/checkpoints/pinn_lbfgs_final.pt
"""

# ── Fix imports regardless of working directory ───────────────────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import numpy as np
import torch
from pathlib import Path

from data.loader             import SU2Loader, WedgeSyntheticData, DataNormalizer
from models.pinn              import ShockwavePINN
from physics.physics_loss     import PhysicsLoss
from training.trainer         import PINNTrainer
from visualization.visualizer import (
    plot_flow_field, plot_mach_contour, plot_training_loss,
    plot_shock_detection, plot_error_field,
)

R_AIR = 287.05
GAMMA = 1.4


def build_parser():
    p = argparse.ArgumentParser(description="Shockwave PINN")
    p.add_argument("--task", choices=["train", "visualize", "train_visualize"],
                   default="train_visualize",
                   help="Select the pipeline stage to run.")
    p.add_argument("--mode", choices=["synthetic", "su2"],
                   default="synthetic",
                   help="Data source for training/inference.")
    p.add_argument("--data_dir", default="path/to/su2/output.csv",
                   help="SU2 CSV data file or directory for mode=su2.")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint file to load for visualization.")
    p.add_argument("--normalizer", default=None,
                   help="Saved normalizer .npz file for inference/visualization.")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--device", default="auto")
    # Flow
    p.add_argument("--mach_inf", type=float, default=2.5)
    p.add_argument("--wedge_angle_deg", type=float, default=15.0,
                   help="Wedge half-angle for the PINN geometry.")
    p.add_argument("--aoa_deg", type=float, default=None,
                   help="Alias for --wedge_angle_deg when the wedge acts like a flow angle of attack.")
    p.add_argument("--p_inf", type=float, default=101325.0)
    p.add_argument("--T_inf", type=float, default=300.0)
    p.add_argument("--rho_inf", type=float, default=1.225)
    p.add_argument("--n_points", type=int, default=8000)
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for synthetic data generation. Use None for non-deterministic samples.")
    # Architecture
    p.add_argument("--hidden_layers", type=int, default=8)
    p.add_argument("--hidden_width", type=int, default=128)
    # Training
    p.add_argument("--n_adam", type=int, default=5000)
    p.add_argument("--n_lbfgs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w_data", type=float, default=1.0)
    p.add_argument("--w_physics", type=float, default=0.1)
    p.add_argument("--w_bc", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=500)
    return p


def parse_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def load_raw_data(args):
    beta_rad = None
    theta_rad = None
    if args.mode == "su2":
        loader = SU2Loader(args.data_dir)
        raw_data = loader.load()
        theta_rad = np.radians(args.wedge_angle_deg)
    elif args.mode == "synthetic":
        synth = WedgeSyntheticData(
            mach_inf             = args.mach_inf,
            wedge_half_angle_deg = args.wedge_angle_deg,
            p_inf                = args.p_inf,
            T_inf                = args.T_inf,
            rho_inf              = args.rho_inf,
            n_points             = args.n_points,
            seed                 = args.seed,
        )
        raw_data = synth.generate()
        beta_rad = synth.beta
        theta_rad = synth.theta
    else:
        raise ValueError("Visualization and training require --mode synthetic or --mode su2.")
    return raw_data, beta_rad, theta_rad


def build_model(args):
    return ShockwavePINN(
        hidden_layers = args.hidden_layers,
        hidden_width  = args.hidden_width,
    )


def build_loss(args):
    return PhysicsLoss(
        w_data    = args.w_data,
        w_physics = args.w_physics,
        w_bc      = args.w_bc,
    )


def build_trainer(model, loss_fn, normalizer, device, output_dir, log_interval):
    return PINNTrainer(
        model          = model,
        loss_fn        = loss_fn,
        normalizer     = normalizer,
        device         = device,
        checkpoint_dir = str(output_dir / "checkpoints"),
        log_interval   = log_interval,
    )


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model


def inference_and_plot(args, model, normalizer, raw_data, beta_rad, theta_rad, outdir, device):
    print("\n[Pipeline] Inference on dense grid ...")
    model.eval()

    res = 250
    x_lin = np.linspace(-1.0, 1.0, res)
    y_lin = np.linspace(-1.0, 1.0, res)
    Xg, Yg = np.meshgrid(x_lin, y_lin)
    x_norm = Xg.ravel()
    y_norm = Yg.ravel()

    x_phys_g = normalizer.inverse_transform_field("x", x_norm)
    y_phys_g = normalizer.inverse_transform_field("y", y_norm)
    theta_r = theta_rad if theta_rad else 0.0
    above = y_phys_g >= x_phys_g * np.tan(theta_r)

    x_q = x_norm[above]
    y_q = y_norm[above]

    pred_norm = model.predict(x_q, y_q, device=device)

    x_q_phys = normalizer.inverse_transform_field("x", x_q)
    y_q_phys = normalizer.inverse_transform_field("y", y_q)

    pinn_dict = {}
    for field in ("rho", "u", "v", "p", "T"):
        pinn_dict[field] = normalizer.inverse_transform_field(field, pred_norm[field])
    a_pred = np.sqrt(GAMMA * R_AIR * np.maximum(pinn_dict["T"], 1.0))
    pinn_dict["Ma"] = np.sqrt(pinn_dict["u"]**2 + pinn_dict["v"]**2) / a_pred

    a_raw = np.sqrt(GAMMA * R_AIR * raw_data.T)
    Ma_raw = np.sqrt(raw_data.u**2 + raw_data.v**2) / a_raw
    cfd_dict = dict(rho=raw_data.rho, u=raw_data.u, v=raw_data.v,
                    p=raw_data.p, T=raw_data.T, Ma=Ma_raw)

    from scipy.interpolate import griddata as gd
    cfd_q = {
        field: gd(
            np.column_stack([raw_data.x, raw_data.y]),
            cfd_dict[field],
            np.column_stack([x_q_phys, y_q_phys]),
            method="linear",
        )
        for field in ("rho", "u", "v", "p", "T", "Ma")
    }

    beta_plot = beta_rad if beta_rad else np.radians(37.0)

    print("[Pipeline] Generating plots ...")
    plot_mach_contour(
        x_q_phys, y_q_phys, pinn_dict["Ma"],
        theta_rad=theta_r, beta_rad=beta_plot,
        mach_inf=args.mach_inf,
        save_path=str(outdir / "mach_contour.png"),
    )
    plot_flow_field(
        x_q_phys, y_q_phys, cfd=cfd_q, pinn=pinn_dict,
        theta_rad=theta_r, beta_rad=beta_plot,
        fields=["rho", "u", "p", "Ma"],
        save_path=str(outdir / "flow_field.png"),
    )
    plot_error_field(
        x_q_phys, y_q_phys, cfd=cfd_q, pinn=pinn_dict,
        theta_rad=theta_r, beta_rad=beta_plot,
        save_path=str(outdir / "error_field.png"),
    )
    plot_shock_detection(
        x_q_phys, y_q_phys,
        rho=pinn_dict["rho"], p=pinn_dict["p"],
        theta_rad=theta_r, beta_rad=beta_plot,
        save_path=str(outdir / "shock_detection.png"),
    )


def main():
    args = build_parser().parse_args()
    if args.aoa_deg is not None:
        args.wedge_angle_deg = args.aoa_deg

    if args.task == "visualize" and args.checkpoint is None:
        raise ValueError("Visualization requires --checkpoint <path>.")
    if args.task == "visualize" and args.mode == "inference":
        raise ValueError("Visualization requires --mode synthetic or --mode su2.")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = parse_device(args.device)
    print(f"\n{'='*60}")
    print(f"  Shockwave PINN  |  task={args.task}  |  mode={args.mode}  |  device={device}")
    print(f"{'='*60}\n")

    raw_data = None
    beta_rad = None
    theta_rad = None
    normalizer = None
    model = build_model(args)

    if args.task in ("train", "train_visualize"):
        raw_data, beta_rad, theta_rad = load_raw_data(args)
        normalizer = DataNormalizer().fit(raw_data)
        norm_data = normalizer.transform(raw_data)
        normalizer.save(str(outdir / "normalizer.npz"))

        if args.checkpoint:
            model = load_checkpoint(model, args.checkpoint, device)
            print(f"[Pipeline] Loaded checkpoint: {args.checkpoint}")

        loss_fn = build_loss(args)
        trainer = build_trainer(model, loss_fn, normalizer, device, outdir,
                                log_interval=max(1, args.n_adam // 20))

        trainer.train(
            norm_data,
            n_adam       = args.n_adam,
            n_lbfgs      = args.n_lbfgs,
            lr           = args.lr,
            theta_rad    = theta_rad,
            warmup_steps = args.warmup_steps,
        )
        plot_training_loss(trainer.history,
                           save_path=str(outdir / "training_loss.png"))
        trainer.save(str(outdir / "model_final.pt"))
        model = trainer.model

    if args.task in ("visualize", "train_visualize"):
        if normalizer is None:
            norm_path = Path(args.normalizer) if args.normalizer else outdir / "normalizer.npz"
            if not norm_path.exists():
                raise FileNotFoundError(
                    f"Normaliser file not found: {norm_path}. "
                    f"Run training first or specify --normalizer <path>."
                )
            normalizer = DataNormalizer.load(str(norm_path))

        if raw_data is None:
            raw_data, beta_rad, theta_rad = load_raw_data(args)

        if args.checkpoint:
            model = load_checkpoint(model, args.checkpoint, device)
        else:
            print("[Pipeline] No checkpoint provided; using model from current session.")

        inference_and_plot(args, model, normalizer, raw_data, beta_rad, theta_rad, outdir, device)

    print(f"\n[Pipeline] Done. Outputs in: {outdir}/")


if __name__ == "__main__":
    main()
