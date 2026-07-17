"""
oblique_shock.py
================
Exact oblique shock solution for supersonic flow over a 2-D wedge.

This script:
  1. Solves the Theta-Beta-Mach relation for the shock angle
  2. Applies Rankine-Hugoniot jump conditions to get post-shock state
  3. Generates a labelled dataset of (x, y) -> (rho, u, v, p, T, Ma)
  4. Plots the flow field and saves the data for PINN training

Run:
    python oblique_shock.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import csv
import os

# ── Output folder ─────────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shock_outputs")
os.makedirs(OUT, exist_ok=True)


# =============================================================================
# SECTION 1 — FREE-STREAM CONDITIONS
# =============================================================================
# Standard sea-level air, Mach 2.5 wind tunnel conditions

GAMMA    = 1.4          # ratio of specific heats for air (dimensionless)
R        = 287.05       # specific gas constant for air [J/(kg·K)]
MACH_INF = 2.5          # free-stream Mach number
T_INF    = 300.0        # free-stream temperature [K]
P_INF    = 101_325.0    # free-stream pressure [Pa]
RHO_INF  = P_INF / (R * T_INF)   # ideal gas: rho = p / (RT) [kg/m³]

THETA_DEG = 15.0                  # wedge half-angle [degrees]
THETA     = np.radians(THETA_DEG) # convert to radians for trig functions

print("=" * 55)
print("  Oblique Shock Solver — 2-D Supersonic Wedge")
print("=" * 55)
print(f"\nFree-stream conditions:")
print(f"  M∞    = {MACH_INF}")
print(f"  p∞    = {P_INF:.0f} Pa")
print(f"  T∞    = {T_INF:.0f} K")
print(f"  ρ∞    = {RHO_INF:.4f} kg/m³")
print(f"  θ     = {THETA_DEG}°  (wedge half-angle)")


# =============================================================================
# SECTION 2 — THETA-BETA-MACH RELATION
# =============================================================================
# For supersonic flow over a wedge of half-angle θ, an oblique shock forms
# at angle β to the free-stream. The relationship between θ, β, and M is:
#
#   tan(θ) = 2·cot(β) · [M²·sin²(β) − 1] / [M²·(γ + cos(2β)) + 2]
#
# This is transcendental — no closed-form solution exists for β.
# We solve it numerically using Brent's root-finding method.
#
# Physical constraints on β:
#   - β > μ = arcsin(1/M)    (Mach angle — minimum possible shock angle)
#   - β < 90°                 (normal shock is the maximum)
#   - We want the WEAK shock solution (lower β, lower loss of total pressure)

def tbm_residual(beta):
    """
    Returns tan(θ) - RHS of the TBM equation.
    Zero when β is the correct shock angle for this Mach and wedge angle.
    """
    numerator   = MACH_INF**2 * np.sin(beta)**2 - 1
    denominator = MACH_INF**2 * (GAMMA + np.cos(2 * beta)) + 2
    rhs = (2 / np.tan(beta)) * (numerator / denominator)
    return np.tan(THETA) - rhs


# Find the bracket for Brent's method
# The weak-shock solution lies between the Mach angle and ~80°
mach_angle = np.arcsin(1.0 / MACH_INF)   # μ = arcsin(1/M)

# Scan the range to find where the residual changes sign
beta_scan = np.linspace(mach_angle + 0.001, np.radians(80), 5000)
residuals  = [tbm_residual(b) for b in beta_scan]

# Find the first sign change (weak shock)
sign_changes = [i for i in range(len(residuals)-1)
                if residuals[i] * residuals[i+1] < 0]

if not sign_changes:
    raise ValueError(
        f"No oblique shock solution found. "
        f"Wedge angle {THETA_DEG}° may exceed the detachment limit for M={MACH_INF}."
    )

# Refine with Brent's method for high accuracy
i    = sign_changes[0]
BETA = brentq(tbm_residual, beta_scan[i], beta_scan[i+1], xtol=1e-12)

print(f"\nTheta-Beta-Mach solution:")
print(f"  Mach angle μ   = {np.degrees(mach_angle):.2f}°")
print(f"  Shock angle β  = {np.degrees(BETA):.4f}°")
print(f"  TBM residual   = {tbm_residual(BETA):.2e}  (should be ~0)")


# =============================================================================
# SECTION 3 — RANKINE-HUGONIOT JUMP CONDITIONS
# =============================================================================
# Across the shock, mass, momentum, and energy are conserved.
# These conservation laws give the Rankine-Hugoniot relations.
#
# The shock-NORMAL Mach number M₁ₙ = M∞·sin(β) drives the jump:
#
#   p₂/p₁   = 1 + 2γ/(γ+1) · (M₁ₙ² - 1)       pressure ratio
#   ρ₂/ρ₁   = (γ+1)·M₁ₙ² / [(γ-1)·M₁ₙ² + 2]   density ratio
#   T₂/T₁   = (p₂/p₁) / (ρ₂/ρ₁)                temperature ratio (ideal gas)
#
# The post-shock Mach number uses the normal component:
#   M₂ₙ = sqrt([(γ-1)·M₁ₙ² + 2] / [2γ·M₁ₙ² - (γ-1)])
#   M₂   = M₂ₙ / sin(β - θ)   (flow is deflected by θ toward wedge)

M1n = MACH_INF * np.sin(BETA)   # normal component of upstream Mach

# Ratios across the shock
p_ratio   = 1 + (2 * GAMMA / (GAMMA + 1)) * (M1n**2 - 1)
rho_ratio = (GAMMA + 1) * M1n**2 / ((GAMMA - 1) * M1n**2 + 2)
T_ratio   = p_ratio / rho_ratio

# Post-shock state (region 2)
P2   = P_INF   * p_ratio
RHO2 = RHO_INF * rho_ratio
T2   = T_INF   * T_ratio

# Post-shock Mach number
M2n = np.sqrt(((GAMMA - 1) * M1n**2 + 2) / (2 * GAMMA * M1n**2 - (GAMMA - 1)))
M2  = M2n / np.sin(BETA - THETA)

print(f"\nRankine-Hugoniot jump conditions:")
print(f"  M₁ₙ (normal Mach)  = {M1n:.4f}")
print(f"  p₂/p₁              = {p_ratio:.4f}")
print(f"  ρ₂/ρ₁              = {rho_ratio:.4f}")
print(f"  T₂/T₁              = {T_ratio:.4f}")
print(f"\nPost-shock state (region 2):")
print(f"  M₂    = {M2:.4f}")
print(f"  p₂    = {P2:.1f} Pa")
print(f"  T₂    = {T2:.2f} K")
print(f"  ρ₂    = {RHO2:.4f} kg/m³")

# Quick sanity check: total temperature should be conserved (adiabatic)
Tt_inf = T_INF * (1 + (GAMMA-1)/2 * MACH_INF**2)
Tt_2   = T2   * (1 + (GAMMA-1)/2 * M2**2)
print(f"\nSanity check (total temperature):")
print(f"  T0_inf = {Tt_inf:.2f} K")
print(f"  T0_2   = {Tt_2:.2f} K")
print(f"  Match:   {'YES' if abs(Tt_inf - Tt_2) < 1.0 else 'NO — check calculation'}")


# =============================================================================
# SECTION 4 — VELOCITY COMPONENTS
# =============================================================================
# Speed of sound: a = sqrt(γRT)
# Velocity magnitude: |V| = M * a
# Direction:
#   - Pre-shock:  horizontal (0°)
#   - Post-shock: deflected downward by θ (toward wedge surface)

a1 = np.sqrt(GAMMA * R * T_INF)   # pre-shock speed of sound [m/s]
a2 = np.sqrt(GAMMA * R * T2)      # post-shock speed of sound [m/s]

V1 = MACH_INF * a1                # pre-shock speed [m/s]
V2 = M2 * a2                      # post-shock speed [m/s]

# Velocity components
U1 =  V1 * np.cos(0)              # horizontal
V1y = V1 * np.sin(0)              # zero vertical

U2 =  V2 * np.cos(THETA)          # post-shock, deflected toward wedge
V2y = -V2 * np.sin(THETA)         # negative: flow turns downward

print(f"\nVelocity components:")
print(f"  Pre-shock:   u={U1:.1f} m/s, v={V1y:.1f} m/s  (horizontal)")
print(f"  Post-shock:  u={U2:.1f} m/s, v={V2y:.1f} m/s  (deflected {THETA_DEG}° down)")


# =============================================================================
# SECTION 5 — GENERATE DATASET
# =============================================================================
# Scatter random points in the 2-D domain.
# Assign exact pre- or post-shock values based on geometry:
#   - Above shock line y = x·tan(β):   region 1 (pre-shock)
#   - Below shock line, above wedge:   region 2 (post-shock)
#   - Below wedge y = x·tan(θ):        solid — excluded

N_POINTS = 8000
DOMAIN   = (0.0, 2.0, 0.0, 1.2)  # xmin, xmax, ymin, ymax
xmin, xmax, ymin, ymax = DOMAIN

rng = np.random.default_rng(seed=42)

# Oversample and filter out points inside the solid wedge
x_all = rng.uniform(xmin, xmax, N_POINTS * 5)
y_all = rng.uniform(ymin, ymax, N_POINTS * 5)

above_wedge = y_all > x_all * np.tan(THETA) + 1e-4   # above wedge surface
x = x_all[above_wedge][:N_POINTS]
y = y_all[above_wedge][:N_POINTS]

# Classify each point: above shock line → region 1, else region 2
above_shock = y > x * np.tan(BETA)

# Assign flow variables
rho = np.where(above_shock, RHO_INF, RHO2)
u   = np.where(above_shock, U1,      U2)
v   = np.where(above_shock, V1y,     V2y)
p   = np.where(above_shock, P_INF,   P2)
T   = np.where(above_shock, T_INF,   T2)
a   = np.where(above_shock, a1,      a2)
Ma  = np.sqrt(u**2 + v**2) / a

print(f"\nDataset generated:")
print(f"  Total points:     {len(x)}")
print(f"  Pre-shock (R1):   {above_shock.sum()}")
print(f"  Post-shock (R2):  {(~above_shock).sum()}")


# =============================================================================
# SECTION 6 — SAVE DATASET TO CSV
# =============================================================================
# Save as CSV so it can be loaded by the PINN training script

csv_path = os.path.join(OUT, "wedge_flow_data.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y", "rho", "u", "v", "p", "T", "Ma"])
    for i in range(len(x)):
        writer.writerow([
            f"{x[i]:.6f}", f"{y[i]:.6f}",
            f"{rho[i]:.6f}", f"{u[i]:.4f}", f"{v[i]:.4f}",
            f"{p[i]:.2f}", f"{T[i]:.4f}", f"{Ma[i]:.6f}",
        ])

print(f"\nDataset saved: {csv_path}")


# =============================================================================
# SECTION 7 — PLOTS
# =============================================================================

def draw_wedge(ax, color="#555555"):
    """Draw the solid wedge as a filled triangle."""
    from matplotlib.patches import Polygon
    pts = np.array([
        [xmin, ymin],
        [xmax, xmax * np.tan(THETA)],
        [xmax, ymin],
    ])
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec="k", lw=1.0, zorder=5))

def draw_shock(ax, **kwargs):
    """Draw the theoretical shock line."""
    xs = [0, xmax]
    ys = [0, xmax * np.tan(BETA)]
    ax.plot(xs, ys, **kwargs)


# ── Plot 1: Flow variable scatter plots ───────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

fields  = [rho,    u,         v,          p,           T,       Ma]
labels  = ["ρ [kg/m³]", "u [m/s]", "v [m/s]", "p [Pa]", "T [K]", "Mach"]
cmaps   = ["viridis", "RdBu_r", "RdBu_r", "RdYlBu_r", "inferno", "plasma"]

for ax, field, label, cmap in zip(axes, fields, labels, cmaps):
    sc = ax.scatter(x, y, c=field, s=2, cmap=cmap)
    draw_wedge(ax)
    draw_shock(ax, color="white", lw=1.5, ls="--",
               label=f"Shock β={np.degrees(BETA):.1f}°")
    plt.colorbar(sc, ax=ax, label=label, shrink=0.85)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(label); ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")

fig.suptitle(
    f"Exact Oblique Shock Solution — M∞={MACH_INF}, θ={THETA_DEG}°, β={np.degrees(BETA):.1f}°",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
path1 = os.path.join(OUT, "flow_variables.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path1}")


# ── Plot 2: Mach contour (the main result) ────────────────────────────────────

from scipy.interpolate import griddata

xi  = np.linspace(xmin, xmax, 300)
yi  = np.linspace(ymin, ymax, 180)
Xi, Yi = np.meshgrid(xi, yi)
Ma_grid = griddata(np.column_stack([x, y]), Ma, (Xi, Yi), method="linear")

fig, ax = plt.subplots(figsize=(11, 5))
levels = np.linspace(1.5, MACH_INF * 1.05, 60)
cf = ax.contourf(Xi, Yi, Ma_grid, levels=levels, cmap="plasma", extend="both")
cs = ax.contour( Xi, Yi, Ma_grid, levels=np.arange(1.6, MACH_INF+0.2, 0.1),
                 colors="white", linewidths=0.5, alpha=0.5)
ax.clabel(cs, fmt="%.1f", fontsize=7, colors="white")

draw_wedge(ax)
draw_shock(ax, color="cyan", lw=2.5, ls="--",
           label=f"Shock angle β = {np.degrees(BETA):.2f}°")

plt.colorbar(cf, ax=ax, label="Mach number")
ax.set_xlabel("x [m]", fontsize=12)
ax.set_ylabel("y [m]", fontsize=12)
ax.set_title(
    f"Mach Number — M∞={MACH_INF}, θ={THETA_DEG}°\n"
    f"Post-shock: M={M2:.3f}, p={P2:.0f} Pa, T={T2:.1f} K",
    fontsize=11, fontweight="bold"
)
ax.legend(loc="upper right", fontsize=10)
ax.set_aspect("equal")
plt.tight_layout()
path2 = os.path.join(OUT, "mach_contour.png")
plt.savefig(path2, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved: {path2}")


# ── Plot 3: TBM curve (shows the physics behind shock angle selection) ─────────

fig, ax = plt.subplots(figsize=(9, 5))

beta_range = np.linspace(mach_angle + 0.001, np.pi/2 - 0.001, 1000)
tbm_vals   = [tbm_residual(b) for b in beta_range]

ax.plot(np.degrees(beta_range), tbm_vals, "b-", lw=2, label="TBM residual")
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.axvline(np.degrees(BETA), color="red", lw=2, ls="--",
           label=f"Weak shock β={np.degrees(BETA):.2f}°")
ax.axvline(np.degrees(mach_angle), color="green", lw=1.5, ls=":",
           label=f"Mach angle μ={np.degrees(mach_angle):.2f}°")

ax.set_xlabel("Shock angle β [degrees]", fontsize=12)
ax.set_ylabel("TBM residual tan(θ) − RHS", fontsize=12)
ax.set_title(f"Theta-Beta-Mach Relation  (M={MACH_INF}, θ={THETA_DEG}°)",
             fontsize=12, fontweight="bold")
ax.set_xlim(20, 90); ax.set_ylim(-0.5, 0.8)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
path3 = os.path.join(OUT, "tbm_curve.png")
plt.savefig(path3, dpi=150)
plt.close()
print(f"Saved: {path3}")


# =============================================================================
# SECTION 8 — PRINT SUMMARY TABLE
# =============================================================================

print(f"\n{'='*55}")
print(f"  Summary")
print(f"{'='*55}")
print(f"  {'Quantity':<25} {'Pre-shock':>12} {'Post-shock':>12}")
print(f"  {'-'*50}")
print(f"  {'Mach number':<25} {MACH_INF:>12.3f} {M2:>12.3f}")
print(f"  {'Pressure [Pa]':<25} {P_INF:>12.1f} {P2:>12.1f}")
print(f"  {'Temperature [K]':<25} {T_INF:>12.1f} {T2:>12.1f}")
print(f"  {'Density [kg/m³]':<25} {RHO_INF:>12.4f} {RHO2:>12.4f}")
print(f"  {'u-velocity [m/s]':<25} {U1:>12.1f} {U2:>12.1f}")
print(f"  {'v-velocity [m/s]':<25} {V1y:>12.1f} {V2y:>12.1f}")
print(f"\n  Shock angle β = {np.degrees(BETA):.4f}°")
print(f"  Output folder: {OUT}/")
print(f"{'='*55}\n")