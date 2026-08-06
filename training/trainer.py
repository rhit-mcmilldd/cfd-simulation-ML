"""
training/trainer.py
-------------------
Two-phase PINN training loop:
  Phase 1 — Adam with physics weight ramped from 0 to w_physics
  Phase 2 — L-BFGS fine-tuning for final convergence

Usage
-----
trainer = PINNTrainer(model, loss_fn, normalizer, device="cuda")
trainer.train(norm_data, n_adam=5000, n_lbfgs=300, theta_rad=0.2618)
trainer.save("checkpoints/final.pt")
"""

import torch
import torch.optim as optim
import numpy as np
import time
from pathlib import Path
from typing import Optional

from data.loader import FlowData, DataNormalizer
from models.pinn import ShockwavePINN
from physics.physics_loss import PhysicsLoss


class PINNTrainer:

    def __init__(
        self,
        model:         ShockwavePINN,
        loss_fn:       PhysicsLoss,
        normalizer:    DataNormalizer,
        device:        str = "cpu",
        checkpoint_dir: str = "checkpoints",
        log_interval:  int = 100,
    ):
        self.model      = model.to(device)
        self.loss_fn    = loss_fn
        self.norm       = normalizer
        self.device     = device
        self.ckpt_dir   = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval

        self.history = {
            "step": [], "total": [], "data": [],
            "physics": [], "bc": [], "lr": [],
            "residuals": [],
        }

    # ── Tensor helpers ────────────────────────────────────────────────────

    def _t(self, arr: np.ndarray, grad: bool = False) -> torch.Tensor:
        t = torch.tensor(arr, dtype=torch.float32, device=self.device)
        if grad:
            t.requires_grad_(True)
        return t

    def _make_wall_points(self, x_range, theta_rad: float, n: int = 300):
        """Sample points on the wedge surface y = x·tan(θ)."""
        # x_range is in physical coordinates because the normaliser stats
        # are computed from the physical training data.
        xs_phys = np.linspace(x_range[0] + 1e-4, x_range[1], n, dtype=np.float32)
        ys_phys = xs_phys * np.tan(theta_rad)

        x_lo, x_hi = self.norm.stats["x"]
        y_lo, y_hi = self.norm.stats["y"]
        x_norm = (2.0 * (xs_phys - x_lo) / (x_hi - x_lo) - 1.0).astype(np.float32)
        y_norm = (2.0 * (ys_phys - y_lo) / (y_hi - y_lo) - 1.0).astype(np.float32)
        return self._t(x_norm), self._t(y_norm)

    def _build_tensors(self, data: FlowData, col_fraction: float = 0.4):
        """Split data into training and collocation sets."""
        n     = data.n_points
        n_col = int(n * col_fraction)
        idx   = np.random.permutation(n)
        d_idx = idx[n_col:]
        c_idx = idx[:n_col]

        def sel(arr, i): return arr[i]
        def to_f(arr):   return arr.astype(np.float32)

        x_d = self._t(to_f(sel(data.x, d_idx)))
        y_d = self._t(to_f(sel(data.y, d_idx)))
        target = {
            k: self._t(to_f(sel(getattr(data, k), d_idx)))
            for k in ("rho", "u", "v", "p", "T")
        }
        x_col = self._t(to_f(sel(data.x, c_idx)))
        y_col = self._t(to_f(sel(data.y, c_idx)))

        return x_d, y_d, target, x_col, y_col

    # ── Phase 1: Adam ─────────────────────────────────────────────────────

    def train_adam(
        self,
        data:         FlowData,
        n_steps:      int   = 5000,
        lr:           float = 1e-3,
        theta_rad:    float = None,
        warmup_steps: int   = 500,
    ):
        """Adam training with physics weight warmup."""
        print(f"\n[Trainer] Adam phase: {n_steps} steps, lr={lr}")
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_steps, eta_min=lr * 0.01
        )

        x_d, y_d, target, x_col, y_col = self._build_tensors(data)

        x_wall = y_wall = None
        if theta_rad is not None:
            x_range_phys = (self.norm.stats["x"][0], self.norm.stats["x"][1])
            x_wall, y_wall = self._make_wall_points(x_range_phys, theta_rad)

        w_max = self.loss_fn.w_physics
        t0_start = time.time()
        t0 = t0_start

        for step in range(1, n_steps + 1):
            # Ramp physics weight from a small positive value to w_max over warmup_steps
            w_phys = w_max * max(1e-6, min(1.0, step / max(1, warmup_steps)))

            optimizer.zero_grad()

            loss, breakdown = self.loss_fn.total_loss(
                self.model,
                x_d, y_d, target,
                x_col, y_col,
                self.norm,
                x_wall_norm  = x_wall,
                y_wall_norm  = y_wall,
                theta_rad    = theta_rad,
                physics_weight_override = w_phys,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Log
            self.history["step"].append(step)
            self.history["total"].append(loss.item())
            self.history["data"].append(
                breakdown.get("data", torch.tensor(0.)).item()
                if isinstance(breakdown.get("data"), torch.Tensor)
                else breakdown.get("data", 0.0))
            self.history["physics"].append(
                breakdown.get("physics", torch.tensor(0.)).item()
                if isinstance(breakdown.get("physics"), torch.Tensor)
                else breakdown.get("physics", 0.0))
            self.history["bc"].append(
                breakdown.get("bc", torch.tensor(0.)).item()
                if isinstance(breakdown.get("bc"), torch.Tensor)
                else breakdown.get("bc", 0.0))
            self.history["lr"].append(scheduler.get_last_lr()[0])
            self.history["residuals"].append(breakdown.get("residuals", {}))

            if step % self.log_interval == 0:
                elapsed = time.time() - t0
                bd = {k: v.item() if isinstance(v, torch.Tensor) else v
                      for k, v in breakdown.items() if k != "residuals"}
                parts = "  ".join(f"{k}={v:.3e}" for k, v in bd.items())
                print(f"  [Adam {step:5d}/{n_steps}]  {parts}  "
                      f"w_phys={w_phys:.3f}  ({elapsed:.1f}s)")
                t0 = time.time()

            if step % 1000 == 0:
                self._save_checkpoint(f"adam_step{step:06d}")

        total_adam = time.time() - t0_start
        print(f"[Trainer] Adam phase complete: {n_steps} steps in {total_adam:.1f}s")

    # ── Phase 2: L-BFGS ──────────────────────────────────────────────────

    def fine_tune_lbfgs(
        self,
        data:      FlowData,
        max_iter:  int   = 300,
        lr:        float = 0.1,
        theta_rad: float = None,
    ):
        """L-BFGS fine-tuning for final precision."""
        print(f"\n[Trainer] L-BFGS phase: max {max_iter} iterations")

        optimizer = optim.LBFGS(
            self.model.parameters(),
            lr=lr,
            max_iter=max_iter,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        x_d, y_d, target, x_col, y_col = self._build_tensors(data)
        x_wall = y_wall = None
        if theta_rad is not None:
            x_range_phys = (self.norm.stats["x"][0], self.norm.stats["x"][1])
            x_wall, y_wall = self._make_wall_points(x_range_phys, theta_rad)

        step_count = [0]
        t0 = time.time()

        def closure():
            optimizer.zero_grad()
            loss, breakdown = self.loss_fn.total_loss(
                self.model,
                x_d, y_d, target,
                x_col, y_col,
                self.norm,
                x_wall_norm=x_wall,
                y_wall_norm=y_wall,
                theta_rad=theta_rad,
            )
            loss.backward()
            step_count[0] += 1
            if step_count[0] % 25 == 0:
                print(f"  [L-BFGS iter {step_count[0]:4d}]  loss={loss.item():.4e}")
            return loss

        optimizer.step(closure)
        elapsed = time.time() - t0
        print(f"[Trainer] L-BFGS done. {step_count[0]} function evaluations in {elapsed:.1f}s.")
        self._save_checkpoint("lbfgs_final")

    # ── Full training pipeline ────────────────────────────────────────────

    def train(
        self,
        data:         FlowData,
        n_adam:       int   = 5000,
        n_lbfgs:      int   = 300,
        lr:           float = 1e-3,
        theta_rad:    float = None,
        warmup_steps: int   = 500,
    ):
        """Run Adam then L-BFGS."""
        start = time.time()
        self.train_adam(data, n_adam, lr, theta_rad, warmup_steps)
        if n_lbfgs > 0:
            self.fine_tune_lbfgs(data, n_lbfgs, theta_rad=theta_rad)
        total = time.time() - start
        print(f"[Trainer] Total training time: {total:.1f}s")

    # ── Checkpoint I/O ────────────────────────────────────────────────────

    def _save_checkpoint(self, tag: str):
        path = self.ckpt_dir / f"pinn_{tag}.pt"
        torch.save({
            "model_state": self.model.state_dict(),
            "history":     self.history,
        }, path)
        print(f"  [Checkpoint] Saved: {path}")

    def save(self, path: str):
        torch.save({
            "model_state": self.model.state_dict(),
            "history":     self.history,
        }, path)
        print(f"[Trainer] Model saved: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        if "history" in ckpt:
            self.history = ckpt["history"]
        print(f"[Trainer] Loaded: {path}")
