"""
training/trainer.py
-------------------
Two-phase PINN training loop:
  Phase 1 — Adam with physics weight ramped from 0 to w_physics
  Phase 2 — L-BFGS fine-tuning for final convergence

This version adds:
  - Configurable batch size (col_fraction controls the data/collocation
    split of whatever n_points you generated) and periodic resampling of
    that split so the fixed point cloud gets reshuffled during training
    instead of being frozen for the whole run.
  - Full training-state checkpointing (model + optimizer + scheduler +
    step count + phase) so you can stop and resume without losing
    progress or restarting the LR / physics-weight schedule from zero.

Usage
-----
trainer = PINNTrainer(model, loss_fn, normalizer, device="cuda")
trainer.train(norm_data, n_adam=20000, n_lbfgs=500, theta_rad=0.2618,
              resume=True)
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

    STATE_FILENAME = "training_state.pt"

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

        # Tracks where we are in the two-phase schedule so `train()` can
        # resume without redoing finished work.
        self.state = {
            "adam_step":  0,     # last completed Adam step
            "adam_total": 0,     # n_adam target used for the run in progress
            "lbfgs_done": False,
        }

    # ── Tensor helpers ────────────────────────────────────────────────────

    def _t(self, arr: np.ndarray, grad: bool = False) -> torch.Tensor:
        t = torch.tensor(arr, dtype=torch.float32, device=self.device)
        if grad:
            t.requires_grad_(True)
        return t

    def _make_wall_points(self, x_range, theta_rad: float, n: int = 300):
        """Sample points on the wedge surface y = x·tan(θ)."""
        xs_phys = np.linspace(x_range[0] + 1e-4, x_range[1], n, dtype=np.float32)
        ys_phys = xs_phys * np.tan(theta_rad)

        x_lo, x_hi = self.norm.stats["x"]
        y_lo, y_hi = self.norm.stats["y"]
        x_norm = (2.0 * (xs_phys - x_lo) / (x_hi - x_lo) - 1.0).astype(np.float32)
        y_norm = (2.0 * (ys_phys - y_lo) / (y_hi - y_lo) - 1.0).astype(np.float32)
        return self._t(x_norm), self._t(y_norm)

    def _build_tensors(self, data: FlowData, col_fraction: float = 0.4):
        """
        Split data into training and collocation sets.

        col_fraction controls the "batch size" split of the fixed point
        cloud you generated (via --n_points): a higher value gives more
        collocation points for the physics residual at the cost of fewer
        labelled data points, and vice versa. Call this again (or use
        resample_interval in train_adam) to redraw a different random
        split from the same underlying point cloud.
        """
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
        data:               FlowData,
        n_steps:            int   = 5000,
        lr:                 float = 1e-3,
        theta_rad:          float = None,
        warmup_steps:       int   = 500,
        col_fraction:       float = 0.4,
        resample_interval:  int   = 0,
        start_step:         int   = 0,
        optimizer_state:    Optional[dict] = None,
        scheduler_state:    Optional[dict] = None,
        checkpoint_every:   int   = 0,
    ):
        """
        Adam training with physics weight warmup.

        Parameters added for resuming / larger batches
        ------------------------------------------------
        col_fraction      : fraction of the point cloud used as collocation
                             points each split (rest is labelled data).
        resample_interval : if > 0, redraw the data/collocation split (and
                             wall points) from the full point cloud every
                             this many steps, so training sees more of the
                             combinatorial variety in a fixed dataset.
        start_step         : resume from this step instead of 1 (keeps the
                              cosine LR schedule and physics-weight ramp
                              continuous across runs).
        optimizer_state /
        scheduler_state     : state dicts to resume Adam + the LR scheduler
                               exactly where a previous run left off.
        checkpoint_every    : if > 0, write a full resumable training-state
                               checkpoint every this many steps (in addition
                               to the periodic model-only snapshots).
        """
        print(f"\n[Trainer] Adam phase: target {n_steps} steps "
              f"(resuming from step {start_step})  lr={lr}")

        if start_step >= n_steps:
            print(f"[Trainer] Adam already at/above target step "
                  f"({start_step} >= {n_steps}); nothing to do.")
            self.state["adam_step"]  = start_step
            self.state["adam_total"] = n_steps
            return

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_steps, eta_min=lr * 0.01
        )

        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            print("[Trainer] Restored Adam optimizer state.")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
            print("[Trainer] Restored LR scheduler state.")
        elif start_step > 0:
            # No saved scheduler state but we know how far we got — fast
            # forward the schedule so the LR curve stays continuous.
            for _ in range(start_step):
                scheduler.step()

        x_d, y_d, target, x_col, y_col = self._build_tensors(data, col_fraction)

        x_wall = y_wall = None
        x_range_phys = None
        if theta_rad is not None:
            x_range_phys = (self.norm.stats["x"][0], self.norm.stats["x"][1])
            x_wall, y_wall = self._make_wall_points(x_range_phys, theta_rad)

        w_max = self.loss_fn.w_physics
        t0_start = time.time()
        t0 = t0_start

        for step in range(start_step + 1, n_steps + 1):
            if resample_interval and step % resample_interval == 0:
                x_d, y_d, target, x_col, y_col = self._build_tensors(data, col_fraction)
                if theta_rad is not None:
                    x_wall, y_wall = self._make_wall_points(x_range_phys, theta_rad)

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

            self.state["adam_step"]  = step
            self.state["adam_total"] = n_steps

            if step % self.log_interval == 0:
                elapsed = time.time() - t0
                bd = {k: v.item() if isinstance(v, torch.Tensor) else v
                      for k, v in breakdown.items() if k != "residuals"}
                parts = "  ".join(f"{k}={v:.3e}" for k, v in bd.items())
                print(f"  [Adam {step:6d}/{n_steps}]  {parts}  "
                      f"w_phys={w_phys:.3f}  ({elapsed:.1f}s)")
                t0 = time.time()

            if step % 1000 == 0:
                self._save_checkpoint(f"adam_step{step:06d}")

            if checkpoint_every and step % checkpoint_every == 0:
                self.save_training_state(optimizer=optimizer, scheduler=scheduler)

        # Always leave a fresh resumable snapshot at the end of the phase.
        self.save_training_state(optimizer=optimizer, scheduler=scheduler)

        total_adam = time.time() - t0_start
        print(f"[Trainer] Adam phase complete: "
              f"{n_steps - start_step} new steps in {total_adam:.1f}s "
              f"(now at step {n_steps}/{n_steps})")

    # ── Phase 2: L-BFGS ──────────────────────────────────────────────────

    def fine_tune_lbfgs(
        self,
        data:         FlowData,
        max_iter:     int   = 300,
        lr:           float = 0.1,
        theta_rad:    float = None,
        col_fraction: float = 0.4,
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

        x_d, y_d, target, x_col, y_col = self._build_tensors(data, col_fraction)
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
        self.state["lbfgs_done"] = True
        self.save_training_state()

    # ── Full training pipeline ────────────────────────────────────────────

    def train(
        self,
        data:               FlowData,
        n_adam:             int   = 5000,
        n_lbfgs:            int   = 300,
        lr:                 float = 1e-3,
        theta_rad:          float = None,
        warmup_steps:       int   = 500,
        col_fraction:       float = 0.4,
        resample_interval:  int   = 0,
        resume:             bool  = False,
        checkpoint_every:   int   = 0,
    ):
        """
        Run Adam then L-BFGS.

        If resume=True, looks for a training_state.pt in checkpoint_dir and,
        if found, restores the model/optimizer/scheduler and continues from
        the saved step instead of starting over. If Adam already reached
        n_adam and L-BFGS already finished, both phases are skipped (no-op
        re-run). If you raise n_adam above what was previously trained, Adam
        resumes and trains the extra steps; L-BFGS then reruns on the
        updated model.
        """
        start = time.time()

        start_step      = 0
        optimizer_state = None
        scheduler_state = None

        if resume:
            state_path = self.ckpt_dir / self.STATE_FILENAME
            if state_path.exists():
                loaded = self.load_training_state(state_path)
                # load_training_state() already updated self.state from the
                # checkpoint's nested "state" dict — read the step from there,
                # not from the top level of the raw checkpoint dict.
                start_step      = self.state.get("adam_step", 0)
                optimizer_state = loaded.get("optimizer_state")
                scheduler_state = loaded.get("scheduler_state")
                print(f"[Trainer] Resume: found checkpoint at Adam step "
                      f"{start_step}, lbfgs_done={self.state['lbfgs_done']}.")
                if start_step >= n_adam:
                    print(f"[Trainer] Requested n_adam={n_adam} <= saved step "
                          f"{start_step}; Adam phase will be skipped. Raise "
                          f"--n_adam to train further.")
            else:
                print(f"[Trainer] --resume set but no {self.STATE_FILENAME} "
                      f"found in {self.ckpt_dir}; starting fresh.")

        self.train_adam(
            data, n_adam, lr, theta_rad, warmup_steps,
            col_fraction      = col_fraction,
            resample_interval = resample_interval,
            start_step        = start_step,
            optimizer_state    = optimizer_state,
            scheduler_state    = scheduler_state,
            checkpoint_every   = checkpoint_every,
        )

        if n_lbfgs > 0:
            if self.state.get("lbfgs_done") and resume:
                print("[Trainer] L-BFGS already completed for this checkpoint; "
                      "skipping. Delete training_state.pt to force a rerun.")
            else:
                self.fine_tune_lbfgs(
                    data, n_lbfgs, theta_rad=theta_rad, col_fraction=col_fraction
                )

        total = time.time() - start
        print(f"[Trainer] Total training time this run: {total:.1f}s")

    # ── Checkpoint I/O ────────────────────────────────────────────────────

    def _save_checkpoint(self, tag: str):
        """Lightweight, model-only snapshot (kept for backwards compatibility)."""
        path = self.ckpt_dir / f"pinn_{tag}.pt"
        torch.save({
            "model_state": self.model.state_dict(),
            "history":     self.history,
        }, path)
        print(f"  [Checkpoint] Saved: {path}")

    def save_training_state(self, optimizer=None, scheduler=None):
        """
        Save everything needed to resume training exactly where it left off:
        model weights, optimizer state, LR scheduler state, step count, and
        which phase has completed. Overwrites the previous state file so
        resuming always picks up the latest progress.
        """
        path = self.ckpt_dir / self.STATE_FILENAME
        torch.save({
            "model_state":     self.model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "history":         self.history,
            "state":           self.state,
        }, path)
        print(f"  [Checkpoint] Saved resumable training state: {path} "
              f"(adam_step={self.state['adam_step']}, "
              f"lbfgs_done={self.state['lbfgs_done']})")

    def load_training_state(self, path=None) -> dict:
        """
        Load a resumable training-state checkpoint, restoring the model
        weights, history, and phase bookkeeping into this trainer. Returns
        the raw dict so the caller can pull out optimizer/scheduler state
        for train_adam.
        """
        path = Path(path) if path is not None else self.ckpt_dir / self.STATE_FILENAME
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        if ckpt.get("history"):
            self.history = ckpt["history"]
        if ckpt.get("state"):
            self.state = ckpt["state"]
        print(f"[Trainer] Loaded resumable training state: {path}")
        return ckpt

    def save(self, path: str):
        """Final model-only save (weights + history), for deployment/inference."""
        torch.save({
            "model_state": self.model.state_dict(),
            "history":     self.history,
        }, path)
        print(f"[Trainer] Model saved: {path}")

    def load(self, path: str):
        """Load model weights + history only (no optimizer state)."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        if "history" in ckpt:
            self.history = ckpt["history"]
        print(f"[Trainer] Loaded: {path}")