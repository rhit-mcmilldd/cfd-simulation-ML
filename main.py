"""
main.py
-------
Shockwave PINN — end-to-end pipeline.

Usage
-----
# Synthetic data (no OpenFOAM needed):
    python main.py --mode synthetic

# Real OpenFOAM data:
    python main.py --mode openfoam --data_dir postProcessing/surfaces

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

from data.openfoam_loader    import OpenFOAMLoader, WedgeSyntheticData, DataNormalizer
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
    p.add_argument("--mode",         choices=["synthetic", "openfoam", "inference"],
                   default="synthetic")
    p.add_argument("--data_dir",     default="postProcessing/surfaces")
    p.add_argument("--checkpoint",   default=None)
    p.add_argument("--output_dir",   default="outputs")
    p.add_argument("--device",       default="auto")
    # Flow
    p.add_argument("--mach_inf",         type=float, default=2.5)
    p.add_argument("--wedge_angle_deg",  type=float, default=15.0)
    p.add_argument("--p_inf",            type=float, default=101325.0)
    p.add_argument("--T_inf",            type=float, default=300.0)
    p.add_argument("--rho_inf",          type=float, default=1.225)
    p.add_argument("--n_points",         type=int,   default=8000)
    # Architecture
    p.add_argument("--hidden_layers",    type=int,   default=8)
    p.add_argument("--hidden_width",     type=int,   default=128)
    # Training
    p.add_argument("--n_adam",           type=int,   default=5000)
    p.add_argument("--n_lbfgs",          type=int,   default=300)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--w_data",           type=float, default=1.0)
    p.add_argument("--w_physics",        type=float, default=0.1)
    p.add_argument("--w_bc",             type=float, default=1.0)
    p.add_argument("--warmup_steps",     type=int,   default=500)
    return p


def main():
    args   = build_parser().parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  Shockwave PINN  |  mode={args.mode}  |  device={device}")
    print(f"{'='*60}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────
    beta_rad  = None
    theta_rad = None

    if args.mode == "openfoam":
        loader   = OpenFOAMLoader(args.data_dir)
        raw_data = loader.load_latest()
        # For OpenFOAM data we still need theta for wall BC
        theta_rad = np.radians(args.wedge_angle_deg)
    else:
        synth = WedgeSyntheticData(
            mach_inf             = args.mach_inf,
            wedge_half_angle_deg = args.wedge_angle_deg,
            p_inf                = args.p_inf,
            T_inf                = args.T_inf,
            rho_inf              = args.rho_inf,
            n_points             = args.n_points,
        )
        raw_data  = synth.generate()
        beta_rad  = synth.beta
        theta_rad = synth.theta

    # Derived Mach for visualisation
    a_raw   = np.sqrt(GAMMA * R_AIR * raw_data.T)
    Ma_raw  = np.sqrt(raw_data.u**2 + raw_data.v**2) / a_raw
    cfd_dict = dict(rho=raw_data.rho, u=raw_data.u, v=raw_data.v,
                    p=raw_data.p, T=raw_data.T, Ma=Ma_raw)

    # ── 2. Normalise ──────────────────────────────────────────────────────
    normalizer = DataNormalizer().fit(raw_data)
    norm_data  = normalizer.transform(raw_data)
    normalizer.save(str(outdir / "normalizer.npz"))

    # ── 3. Model ──────────────────────────────────────────────────────────
    model = ShockwavePINN(
        hidden_layers = args.hidden_layers,
        hidden_width  = args.hidden_width,
    )
    print(f"[Pipeline] Model: {model.n_parameters():,} parameters")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[Pipeline] Loaded checkpoint: {args.checkpoint}")

    # ── 4. Loss function ──────────────────────────────────────────────────
    loss_fn = PhysicsLoss(
        w_data    = args.w_data,
        w_physics = args.w_physics,
        w_bc      = args.w_bc,
    )

    # ── 5. Train ──────────────────────────────────────────────────────────
    trainer = PINNTrainer(
        model          = model,
        loss_fn        = loss_fn,
        normalizer     = normalizer,
        device         = device,
        checkpoint_dir = str(outdir / "checkpoints"),
        log_interval   = max(1, args.n_adam // 20),
    )

    if args.mode != "inference":
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

    # ── 6. Inference grid ─────────────────────────────────────────────────
    print("\n[Pipeline] Inference on dense grid ...")
    model.eval()

    res   = 250
    x_lin = np.linspace(norm_data.x.min(), norm_data.x.max(), res)
    y_lin = np.linspace(norm_data.y.min(), norm_data.y.max(), res)
    Xg, Yg = np.meshgrid(x_lin, y_lin)

    x_phys_g = normalizer.inverse_transform_field("x", Xg.ravel())
    y_phys_g = normalizer.inverse_transform_field("y", Yg.ravel())
    theta_r  = theta_rad if theta_rad else 0.0
    above    = y_phys_g >= x_phys_g * np.tan(theta_r)

    x_q = Xg.ravel()[above]
    y_q = Yg.ravel()[above]

    pred_norm = model.predict(x_q, y_q, device=device)

    pinn_dict = {}
    for field in ("rho", "u", "v", "p", "T"):
        pinn_dict[field] = normalizer.inverse_transform_field(field, pred_norm[field])
    a_pred = np.sqrt(GAMMA * R_AIR * np.maximum(pinn_dict["T"], 1.0))
    pinn_dict["Ma"] = np.sqrt(pinn_dict["u"]**2 + pinn_dict["v"]**2) / a_pred

    x_q_phys = normalizer.inverse_transform_field("x", x_q)
    y_q_phys = normalizer.inverse_transform_field("y", y_q)

    # Interpolate CFD reference to query points
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

    # ── 7. Plots ──────────────────────────────────────────────────────────
    print("[Pipeline] Generating plots ...")
    beta_plot = beta_rad if beta_rad else np.radians(37.0)

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

    print(f"\n[Pipeline] Done. Outputs in: {outdir}/")


if __name__ == "__main__":
    main()
