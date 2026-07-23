"""
physics/physics_loss.py
-----------------------
Embeds the 2-D steady Euler equations into the PINN training loss using
automatic differentiation. No finite differences, no mesh — derivatives
are computed exactly through the network via torch.autograd.

Equations (strong conservation form):
    ∂(ρu)/∂x + ∂(ρv)/∂y = 0                      [continuity]
    ∂(ρu²+p)/∂x + ∂(ρuv)/∂y = 0                  [x-momentum]
    ∂(ρuv)/∂x + ∂(ρv²+p)/∂y = 0                  [y-momentum]
    p - ρRT = 0                                     [ideal gas]

Key implementation detail
-------------------------
The network takes NORMALISED inputs (x_norm, y_norm) in [-1,1] and returns
NORMALISED outputs. To evaluate the PDE in physical units we must:
  1. Keep the network outputs in the computation graph (no .detach())
  2. Denormalise using in-graph arithmetic: x_phys = (x_norm+1)/2 * span + lo
  3. Apply the chain rule for spatial derivatives:
        ∂f/∂x_phys = (∂f/∂x_norm) * (1 / span_x)
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict

GAMMA = 1.4
R_AIR = 287.05  # J/(kg·K)


# ── Autograd derivative helpers ──────────────────────────────────────────────

def _grad(output: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
    """Compute ∂(sum(output))/∂coord via autograd."""
    g = torch.autograd.grad(
        output.sum(), coord,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    return g if g is not None else torch.zeros_like(coord)


# ── Euler residuals ──────────────────────────────────────────────────────────

def euler_residuals(
    x_norm:   torch.Tensor,   # (N,)  normalised x, requires_grad=True
    y_norm:   torch.Tensor,   # (N,)  normalised y, requires_grad=True
    rho:      torch.Tensor,   # (N,)  physical density
    u:        torch.Tensor,   # (N,)  physical x-velocity
    v:        torch.Tensor,   # (N,)  physical y-velocity
    p:        torch.Tensor,   # (N,)  physical pressure
    T:        torch.Tensor,   # (N,)  physical temperature
    span_x:   float,          # physical x span (for chain rule)
    span_y:   float,          # physical y span (for chain rule)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (R_cont, R_xmom, R_ymom, R_eos) — each (N,).
    All residuals should be zero for a correct flow solution.
    """

    def ddx(f):
        return _grad(f, x_norm) / span_x

    def ddy(f):
        return _grad(f, y_norm) / span_y

    R_cont = ddx(rho * u) + ddy(rho * v)
    R_xmom = ddx(rho * u**2 + p) + ddy(rho * u * v)
    R_ymom = ddx(rho * u * v) + ddy(rho * v**2 + p)
    R_eos  = p - rho * R_AIR * torch.clamp(T, min=1.0)

    return R_cont, R_xmom, R_ymom, R_eos


# ── Main physics loss class ──────────────────────────────────────────────────

class PhysicsLoss(nn.Module):
    """
    Combined loss: data fidelity + PDE residuals + boundary conditions.

    Total loss = w_data * L_data + w_physics * L_physics + w_bc * L_bc

    Parameters
    ----------
    w_data    : weight for data/CFD matching loss
    w_physics : weight for Euler equation residuals
    w_bc      : weight for wall no-penetration BC
    gamma     : ratio of specific heats
    R         : specific gas constant [J/(kg·K)]
    """

    def __init__(
        self,
        w_data:    float = 1.0,
        w_physics: float = 0.1,
        w_bc:      float = 1.0,
        gamma:     float = GAMMA,
        R:         float = R_AIR,
    ):
        super().__init__()
        self.w_data    = w_data
        self.w_physics = w_physics
        self.w_bc      = w_bc
        self.gamma     = gamma
        self.R         = R
        self.mse       = nn.MSELoss()

    def data_loss(self, pred: dict, target: dict) -> torch.Tensor:
        """MSE between network predictions and CFD/synthetic reference data."""
        return sum(
            self.mse(pred[k].reshape(-1), target[k].reshape(-1))
            for k in ("rho", "u", "v", "p", "T")
        )

    def physics_loss(
        self,
        model,
        x_col_norm: torch.Tensor,
        y_col_norm: torch.Tensor,
        normalizer,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Evaluate Euler residuals at collocation points.

        Parameters
        ----------
        model       : ShockwavePINN instance
        x_col_norm  : (N,) normalised x collocation points
        y_col_norm  : (N,) normalised y collocation points
        normalizer  : DataNormalizer (for in-graph denormalisation)
        """
        # Detach and re-enable grad so these are leaf tensors for autograd
        x = x_col_norm.detach().requires_grad_(True)
        y = y_col_norm.detach().requires_grad_(True)

        out = model(x, y)

        # Denormalise IN GRAPH (keeps grad_fn connected)
        rho = normalizer.inverse_transform_tensor("rho", out["rho"].squeeze(-1))
        u   = normalizer.inverse_transform_tensor("u",   out["u"].squeeze(-1))
        v   = normalizer.inverse_transform_tensor("v",   out["v"].squeeze(-1))
        p   = normalizer.inverse_transform_tensor("p",   out["p"].squeeze(-1))
        T   = normalizer.inverse_transform_tensor("T",   out["T"].squeeze(-1))

        span_x, span_y = normalizer.physical_coord_scale()

        R_cont, R_xmom, R_ymom, R_eos = euler_residuals(
            x, y, rho, u, v, p, T, span_x, span_y
        )

        # Scale residuals to similar magnitudes before squaring
        rho_scale = rho.detach().mean().clamp(min=1e-6)
        u_scale   = u.detach().abs().mean().clamp(min=1e-6)
        p_scale   = p.detach().mean().clamp(min=1e-6)

        loss = (
            (R_cont / (rho_scale * u_scale)).pow(2).mean()
          + (R_xmom / p_scale).pow(2).mean()
          + (R_ymom / p_scale).pow(2).mean()
          + (R_eos  / p_scale).pow(2).mean()
        )

        residuals = {
            "continuity": R_cont.abs().mean().item(),
            "x_momentum": R_xmom.abs().mean().item(),
            "y_momentum": R_ymom.abs().mean().item(),
            "ideal_gas":  R_eos.abs().mean().item(),
        }

        return loss, residuals

    def bc_loss(
        self,
        pred_wall: dict,
        theta_rad: float,
    ) -> torch.Tensor:
        """
        No-penetration condition on wedge surface: V · n = 0

        Wedge surface: y = x·tan(θ)
        Outward normal: n = (-tan(θ), 1) / ||n||
        """
        import math
        norm_mag = math.sqrt(math.tan(theta_rad)**2 + 1.0)
        nx = -math.tan(theta_rad) / norm_mag
        ny =  1.0 / norm_mag

        u_wall = pred_wall["u"].reshape(-1)
        v_wall = pred_wall["v"].reshape(-1)
        Vn     = u_wall * nx + v_wall * ny
        return Vn.pow(2).mean()

    def total_loss(
        self,
        model,
        x_data_norm:  torch.Tensor,
        y_data_norm:  torch.Tensor,
        target:       dict,
        x_col_norm:   torch.Tensor,
        y_col_norm:   torch.Tensor,
        normalizer,
        x_wall_norm:  torch.Tensor = None,
        y_wall_norm:  torch.Tensor = None,
        theta_rad:    float        = None,
        physics_weight_override: float = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the full combined loss.

        Returns (total_loss, breakdown_dict)
        """
        w_phys = physics_weight_override if physics_weight_override is not None \
                 else self.w_physics

        losses = {}

        # ── Data loss ──────────────────────────────────────────────────
        pred_data = model(x_data_norm, y_data_norm)
        losses["data"] = self.w_data * self.data_loss(pred_data, target)

        # ── Physics loss ───────────────────────────────────────────────
        if w_phys > 0:
            l_phys, residuals = self.physics_loss(
                model, x_col_norm, y_col_norm, normalizer
            )
            losses["physics"] = w_phys * l_phys
        else:
            residuals = {}

        # ── Wall BC loss ───────────────────────────────────────────────
        if x_wall_norm is not None and theta_rad is not None:
            pred_wall = model(x_wall_norm, y_wall_norm)
            # Denormalise velocity for BC (kept in graph)
            pred_wall_phys = {
                "u": normalizer.inverse_transform_tensor(
                    "u", pred_wall["u"].squeeze(-1)),
                "v": normalizer.inverse_transform_tensor(
                    "v", pred_wall["v"].squeeze(-1)),
            }
            losses["bc"] = self.w_bc * self.bc_loss(pred_wall_phys, theta_rad)

        total = sum(losses.values())
        return total, {**losses, "residuals": residuals}
