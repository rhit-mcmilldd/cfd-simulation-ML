"""
physics/physics_loss.py
-----------------------
Embeds the 2-D steady Euler equations into the PINN training loss using
automatic differentiation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict

GAMMA = 1.4
R_AIR = 287.05  # J/(kg·K)


def _grad(output: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
    g = torch.autograd.grad(
        output.sum(), coord,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    return g if g is not None else torch.zeros_like(coord)


def euler_residuals(
    x_norm:   torch.Tensor,
    y_norm:   torch.Tensor,
    rho:      torch.Tensor,
    u:        torch.Tensor,
    v:        torch.Tensor,
    p:        torch.Tensor,
    T:        torch.Tensor,
    span_x:   float,
    span_y:   float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    def ddx(f):
        return _grad(f, x_norm) * (2.0 / span_x)

    def ddy(f):
        return _grad(f, y_norm) * (2.0 / span_y)

    R_cont = ddx(rho * u) + ddy(rho * v)
    R_xmom = ddx(rho * u**2 + p) + ddy(rho * u * v)
    R_ymom = ddx(rho * u * v) + ddy(rho * v**2 + p)
    R_eos  = p - rho * R_AIR * torch.clamp(T, min=1.0)

    return R_cont, R_xmom, R_ymom, R_eos


class PhysicsLoss(nn.Module):
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
        x = x_col_norm.detach().requires_grad_(True)
        y = y_col_norm.detach().requires_grad_(True)

        out = model(x, y)

        rho = normalizer.inverse_transform_tensor("rho", out["rho"].squeeze(-1))
        u   = normalizer.inverse_transform_tensor("u",   out["u"].squeeze(-1))
        v   = normalizer.inverse_transform_tensor("v",   out["v"].squeeze(-1))
        p   = normalizer.inverse_transform_tensor("p",   out["p"].squeeze(-1))
        T   = normalizer.inverse_transform_tensor("T",   out["T"].squeeze(-1))

        span_x, span_y = normalizer.physical_coord_scale()

        R_cont, R_xmom, R_ymom, R_eos = euler_residuals(
            x, y, rho, u, v, p, T, span_x, span_y
        )

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
        nx,
        ny,
    ) -> torch.Tensor:
        """
        Enforce no-penetration by driving wall-normal velocity to zero.
        nx, ny are per-point outward unit normal components (same shape as
        u_wall/v_wall), supporting piecewise surfaces where the normal
        direction varies along the wall (e.g. flat leading edge + wedge).
        """
        u_wall = pred_wall["u"].reshape(-1)
        v_wall = pred_wall["v"].reshape(-1)
        Vn     = u_wall * nx + v_wall * ny
        vel_scale = torch.sqrt((u_wall**2 + v_wall**2).mean()).clamp(min=1e-6)
        return (Vn / vel_scale).pow(2).mean()

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
        nx_wall       = None,
        ny_wall       = None,
        physics_weight_override: float = None,
    ) -> Tuple[torch.Tensor, dict]:
        w_phys = physics_weight_override if physics_weight_override is not None \
                 else self.w_physics

        losses = {}

        pred_data = model(x_data_norm, y_data_norm)
        losses["data"] = self.w_data * self.data_loss(pred_data, target)

        if w_phys > 0:
            l_phys, residuals = self.physics_loss(
                model, x_col_norm, y_col_norm, normalizer
            )
            losses["physics"] = w_phys * l_phys
        else:
            residuals = {}

        if x_wall_norm is not None and nx_wall is not None:
            pred_wall = model(x_wall_norm, y_wall_norm)
            pred_wall_phys = {
                "u": normalizer.inverse_transform_tensor(
                    "u", pred_wall["u"].squeeze(-1)),
                "v": normalizer.inverse_transform_tensor(
                    "v", pred_wall["v"].squeeze(-1)),
            }
            losses["bc"] = self.w_bc * self.bc_loss(pred_wall_phys, nx_wall, ny_wall)

        total = sum(losses.values())
        return total, {**losses, "residuals": residuals}
