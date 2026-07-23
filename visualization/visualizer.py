"""
visualization/visualizer.py
---------------------------
All plotting functions for the shockwave PINN project.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.interpolate import griddata
from pathlib import Path


def _wedge(ax, theta, xmax, color="#555"):
    pts = np.array([[0,0],[xmax, xmax*np.tan(theta)],[xmax,0]])
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec="k", lw=0.8, zorder=5))

def _shock(ax, beta, xmax, **kw):
    ax.plot([0, xmax], [0, xmax*np.tan(beta)], **kw)

def _grid(x, y, f, res=250):
    xi = np.linspace(x.min(), x.max(), res)
    yi = np.linspace(y.min(), y.max(), res)
    Xi, Yi = np.meshgrid(xi, yi)
    Fi = griddata(np.column_stack([x,y]), f, (Xi,Yi), method="linear")
    return Xi, Yi, Fi


def plot_mach_contour(x, y, Ma, theta_rad, beta_rad, mach_inf=2.5,
                      resolution=300, save_path="mach_contour.png"):
    Xi, Yi, Ma_g = _grid(x, y, Ma, resolution)
    fig, ax = plt.subplots(figsize=(10, 5))
    levels = np.linspace(0.5, mach_inf * 1.1, 80)
    cf = ax.contourf(Xi, Yi, Ma_g, levels=levels, cmap="plasma", extend="both")
    cs = ax.contour(Xi, Yi, Ma_g, levels=np.arange(1.0, mach_inf+0.5, 0.25),
                    colors="white", linewidths=0.5, alpha=0.5)
    ax.clabel(cs, fmt="%.2f", fontsize=7, colors="white")
    _shock(ax, beta_rad, Xi[0].max(), color="cyan", lw=2, ls="--",
           label=f"Shock β={np.degrees(beta_rad):.1f}°")
    _wedge(ax, theta_rad, Xi[0].max())
    plt.colorbar(cf, ax=ax, label="Mach number")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"PINN Mach Contour  M∞={mach_inf}  θ={np.degrees(theta_rad):.0f}°",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Vis] Saved: {save_path}")


def plot_flow_field(x, y, cfd, pinn, theta_rad, beta_rad,
                    fields=("rho","u","p","Ma"), resolution=200,
                    save_path="flow_field.png"):
    cmaps = {"rho":"viridis","u":"RdBu_r","v":"RdBu_r",
             "p":"RdYlBu_r","T":"inferno","Ma":"plasma"}
    labels = {"rho":"Density [kg/m³]","u":"u-vel [m/s]","v":"v-vel [m/s]",
              "p":"Pressure [Pa]","T":"Temperature [K]","Ma":"Mach"}

    n = len(fields)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4*n), constrained_layout=True)
    if n == 1: axes = axes[None, :]
    fig.suptitle("CFD  |  PINN  |  Error", fontsize=14, fontweight="bold")

    for row, field in enumerate(fields):
        if field not in cfd or field not in pinn:
            continue
        Xi, Yi, cfd_g  = _grid(x, y, cfd[field],  resolution)
        _,  _,  pinn_g = _grid(x, y, pinn[field], resolution)
        err_g = np.abs(pinn_g - cfd_g)

        xmax = Xi[0].max()
        vmin, vmax = np.nanmin(cfd_g), np.nanmax(cfd_g)
        cmap = cmaps.get(field, "viridis")

        for col, (data_g, title, cm, vn) in enumerate([
            (cfd_g,  "CFD",   cmap,    (vmin, vmax)),
            (pinn_g, "PINN",  cmap,    (vmin, vmax)),
            (err_g,  "Error", "hot_r", (0, np.nanmax(err_g))),
        ]):
            ax = axes[row, col]
            im = ax.contourf(Xi, Yi, data_g, levels=64, cmap=cm,
                             vmin=vn[0], vmax=vn[1], extend="both")
            _wedge(ax, theta_rad, xmax)
            _shock(ax, beta_rad, xmax, color="w", lw=1.2, ls="--")
            plt.colorbar(im, ax=ax, shrink=0.85)
            ax.set_title(f"{title} — {labels.get(field, field)}", fontsize=9)
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
            ax.set_aspect("equal")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Vis] Saved: {save_path}")


def plot_training_loss(history, save_path="training_loss.png"):
    steps = history["step"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    colours = {"total":"#1f77b4","data":"#ff7f0e","physics":"#2ca02c","bc":"#d62728"}
    for k, c in colours.items():
        vals = history.get(k, [])
        if len(vals) == len(steps) and any(v > 0 for v in vals):
            ax1.semilogy(steps, vals, label=k.capitalize(), color=c, lw=1.8)
    ax1.set_xlabel("Step"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss History", fontweight="bold")
    ax1.legend(); ax1.grid(True, which="both", alpha=0.3)

    ax2.semilogy(steps, history["lr"], color="#8c564b", lw=1.5)
    ax2.set_xlabel("Step"); ax2.set_ylabel("Learning rate")
    ax2.set_title("LR Schedule"); ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Vis] Saved: {save_path}")


def plot_error_field(x, y, cfd, pinn, theta_rad, beta_rad,
                     resolution=200, save_path="error_field.png"):
    fields = [k for k in ("rho","u","v","p","T") if k in cfd and k in pinn]
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4), constrained_layout=True)
    if n == 1: axes = [axes]

    for ax, field in zip(axes, fields):
        Xi, Yi, cfd_g  = _grid(x, y, cfd[field],  resolution)
        _,  _,  pinn_g = _grid(x, y, pinn[field], resolution)
        rel = 100 * np.abs(pinn_g - cfd_g) / (np.nanmax(np.abs(cfd_g)) + 1e-12)

        im = ax.contourf(Xi, Yi, rel, levels=50, cmap="hot_r", extend="max")
        _wedge(ax, theta_rad, Xi[0].max())
        _shock(ax, beta_rad, Xi[0].max(), color="cyan", lw=1.2, ls="--")
        plt.colorbar(im, ax=ax, label="Rel. error [%]", shrink=0.85)
        ax.set_title(field); ax.set_xlabel("x [m]"); ax.set_aspect("equal")

    fig.suptitle("PINN Relative Error vs Reference", fontweight="bold")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Vis] Saved: {save_path}")


def plot_shock_detection(x, y, rho, p, theta_rad, beta_rad,
                         resolution=250, save_path="shock_detection.png"):
    from scipy.ndimage import gaussian_gradient_magnitude

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    xmax = x.max()

    for ax, field, label in [(ax1, p, "|∇p|"), (ax2, rho, "|∇ρ|")]:
        Xi, Yi, Fg = _grid(x, y, field, resolution)
        Fg = np.nan_to_num(Fg, nan=np.nanmean(Fg[~np.isnan(Fg)]))
        grad_mag = gaussian_gradient_magnitude(Fg, sigma=2)
        im = ax.contourf(Xi, Yi, grad_mag, levels=80, cmap="hot")
        _shock(ax, beta_rad, xmax, color="cyan", lw=2, ls="--", label="Theory β")
        _wedge(ax, theta_rad, xmax)
        plt.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title(label); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_aspect("equal"); ax.legend(fontsize=8)

    fig.suptitle("Gradient-Based Shock Detection", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Vis] Saved: {save_path}")
