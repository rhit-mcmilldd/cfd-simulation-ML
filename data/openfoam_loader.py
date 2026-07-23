"""
data/openfoam_loader.py
-----------------------
Handles all data ingestion for the Shockwave PINN project:

  1. OpenFOAMLoader  — reads postProcessing surface sample data from a real
                       OpenFOAM rhoCentralFoam / sonicFoam run.
  2. WedgeSyntheticData — generates exact oblique-shock data from the
                          Rankine-Hugoniot jump conditions (no CFD needed).
  3. DataNormalizer  — min-max scales all fields to [-1, 1] for stable training.

OpenFOAM expected layout
------------------------
After running postProcess with a surfaces function object, data appears as:

  postProcessing/
    surfaces/
      {time}/
        wedge_p.raw      # columns: x  y  z  p
        wedge_U.raw      # columns: x  y  z  Ux  Uy  Uz
        wedge_rho.raw    # columns: x  y  z  rho
        wedge_T.raw      # columns: x  y  z  T
        wedge_Ma.raw     # columns: x  y  z  Ma   (optional)

The corresponding OpenFOAM system/controlDict postProcessing entry is
provided in openfoam_case/system/controlDict (see that file).
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from dataclasses import dataclass, fields
from typing import Dict, Tuple, Optional
import torch


# ── Physical constants ──────────────────────────────────────────────────────
R_AIR = 287.05   # J/(kg·K)
GAMMA = 1.4


# ── Data container ──────────────────────────────────────────────────────────

@dataclass
class FlowData:
    """
    Container for a 2-D flow field snapshot.
    All arrays are 1-D with shape (N,).
    """
    x:   np.ndarray   # x-coordinate  [m]
    y:   np.ndarray   # y-coordinate  [m]
    rho: np.ndarray   # density       [kg/m³]
    u:   np.ndarray   # x-velocity    [m/s]
    v:   np.ndarray   # y-velocity    [m/s]
    p:   np.ndarray   # pressure      [Pa]
    T:   np.ndarray   # temperature   [K]
    Ma:  np.ndarray   # Mach number   [-]

    @property
    def n_points(self) -> int:
        return len(self.x)

    def field_names(self):
        return [f.name for f in fields(self)]

    def to_tensors(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        return {
            f.name: torch.tensor(getattr(self, f.name), dtype=torch.float32).to(device)
            for f in fields(self)
        }

    def summary(self):
        print(f"FlowData: {self.n_points} points")
        for f in fields(self):
            arr = getattr(self, f.name)
            print(f"  {f.name:4s}  min={arr.min():.4g}  max={arr.max():.4g}  "
                  f"mean={arr.mean():.4g}")


# ── OpenFOAM reader ─────────────────────────────────────────────────────────

class OpenFOAMLoader:
    """
    Reads postProcessing surface-sample data written by OpenFOAM.

    Supports:
      - .raw files  (space-delimited, # comments)
      - automatic latest-time-directory detection
      - optional Mach field (computed from U and T if not present)

    Parameters
    ----------
    surfaces_dir : path to postProcessing/surfaces  (or equivalent)

    Usage
    -----
    loader = OpenFOAMLoader("postProcessing/surfaces")
    data   = loader.load_latest()
    data.summary()
    """

    FIELD_FILES = {
        "p":   "wedge_p.raw",
        "U":   "wedge_U.raw",
        "rho": "wedge_rho.raw",
        "T":   "wedge_T.raw",
        "Ma":  "wedge_Ma.raw",   # optional
    }

    def __init__(self, surfaces_dir: str):
        self.root = Path(surfaces_dir)
        if not self.root.exists():
            raise FileNotFoundError(
                f"OpenFOAM surfaces directory not found: {self.root}\n"
                f"Expected path: postProcessing/surfaces/\n"
                f"Run OpenFOAM with the surfaces postProcess function object first."
            )

    def _latest_time_dir(self) -> Path:
        """Return the subdirectory with the largest time value."""
        dirs = [d for d in self.root.iterdir() if d.is_dir()]
        if not dirs:
            raise RuntimeError(f"No time directories found in {self.root}")
        return max(dirs, key=lambda d: float(d.name))

    def _read_raw(self, filepath: Path) -> np.ndarray:
        """
        Parse an OpenFOAM .raw surface file.
        Lines starting with # are comments; blank lines are skipped.
        Returns float array of shape (N, n_cols).
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Expected OpenFOAM file not found: {filepath}")
        rows = []
        with open(filepath, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    rows.append([float(v) for v in line.split()])
                except ValueError:
                    continue   # skip malformed lines
        if not rows:
            raise RuntimeError(f"No data parsed from {filepath}")
        return np.array(rows, dtype=np.float64)

    def load_latest(self) -> FlowData:
        """Load data from the latest time directory."""
        t_dir = self._latest_time_dir()
        print(f"[OpenFOAMLoader] Loading time = {t_dir.name}")
        return self._load_from(t_dir)

    def load_time(self, time: float) -> FlowData:
        """Load data from a specific time directory."""
        t_dir = self.root / str(time)
        if not t_dir.exists():
            raise FileNotFoundError(f"Time directory not found: {t_dir}")
        return self._load_from(t_dir)

    def _load_from(self, t_dir: Path) -> FlowData:
        p_raw   = self._read_raw(t_dir / self.FIELD_FILES["p"])
        U_raw   = self._read_raw(t_dir / self.FIELD_FILES["U"])
        rho_raw = self._read_raw(t_dir / self.FIELD_FILES["rho"])
        T_raw   = self._read_raw(t_dir / self.FIELD_FILES["T"])

        x   = p_raw[:, 0].astype(np.float32)
        y   = p_raw[:, 1].astype(np.float32)
        p   = p_raw[:, 3].astype(np.float32)
        u   = U_raw[:, 3].astype(np.float32)
        v   = U_raw[:, 4].astype(np.float32)
        rho = rho_raw[:, 3].astype(np.float32)
        T   = T_raw[:, 3].astype(np.float32)

        # Mach: use file if present, otherwise compute from U and T
        ma_path = t_dir / self.FIELD_FILES["Ma"]
        if ma_path.exists():
            Ma_raw = self._read_raw(ma_path)
            Ma = Ma_raw[:, 3].astype(np.float32)
        else:
            a  = np.sqrt(GAMMA * R_AIR * np.maximum(T, 1.0))
            Ma = np.sqrt(u**2 + v**2) / a
            print("[OpenFOAMLoader] Mach field not found — computed from U and T.")

        n = len(x)
        print(f"[OpenFOAMLoader] Loaded {n} surface points.")

        data = FlowData(x=x, y=y, rho=rho, u=u, v=v, p=p, T=T, Ma=Ma)
        data.summary()
        return data

    def available_times(self):
        """List all available time directories."""
        return sorted(
            [float(d.name) for d in self.root.iterdir() if d.is_dir()],
        )


# ── Synthetic data generator ─────────────────────────────────────────────────

class WedgeSyntheticData:
    """
    Generates an exact oblique-shock flow field using the Theta-Beta-Mach
    relation and Rankine-Hugoniot jump conditions.

    This is used for:
      - Prototyping the PINN before OpenFOAM runs complete
      - Validating the PINN against a known analytical solution
      - Unit-testing the physics loss (residuals should be near zero)

    Reference: Anderson, J.D. (2003) Modern Compressible Flow, 3rd ed., §4.3

    Parameters
    ----------
    mach_inf             : free-stream Mach number (must be > 1)
    wedge_half_angle_deg : wedge half-angle in degrees
    gamma                : ratio of specific heats (1.4 for air)
    p_inf                : free-stream pressure [Pa]
    T_inf                : free-stream temperature [K]
    rho_inf              : free-stream density [kg/m³]
    domain               : (xmin, xmax, ymin, ymax)
    n_points             : number of training points to generate
    """

    def __init__(
        self,
        mach_inf:             float = 2.5,
        wedge_half_angle_deg: float = 15.0,
        gamma:                float = 1.4,
        p_inf:                float = 101_325.0,
        T_inf:                float = 300.0,
        rho_inf:              float = 1.225,
        domain:               Tuple = (0.0, 2.0, 0.0, 1.2),
        n_points:             int   = 8000,
    ):
        if mach_inf <= 1.0:
            raise ValueError(f"Mach number must be > 1 for supersonic flow, got {mach_inf}")

        self.M1     = mach_inf
        self.theta  = np.radians(wedge_half_angle_deg)
        self.gamma  = gamma
        self.p1     = p_inf
        self.T1     = T_inf
        self.rho1   = rho_inf
        self.domain = domain
        self.n_pts  = n_points
        self.R      = R_AIR

        self.beta = self._solve_shock_angle()
        self._post = self._rankine_hugoniot()

        print(f"[WedgeSynth] M∞={self.M1}  θ={wedge_half_angle_deg:.1f}°  "
              f"β={np.degrees(self.beta):.2f}°")
        print(f"[WedgeSynth] Post-shock: M={self._post['M']:.3f}  "
              f"p={self._post['p']:.0f} Pa  T={self._post['T']:.1f} K  "
              f"ρ={self._post['rho']:.4f} kg/m³")

    # ── Theta-Beta-Mach solver ────────────────────────────────────────────

    def _tbm_residual(self, beta: float) -> float:
        M, theta, g = self.M1, self.theta, self.gamma
        num = M**2 * np.sin(beta)**2 - 1
        den = M**2 * (g + np.cos(2 * beta)) + 2
        return np.tan(theta) - (2 / np.tan(beta)) * (num / den)

    def _solve_shock_angle(self) -> float:
        from scipy.optimize import brentq
        # Physical bound: β > Mach angle μ = arcsin(1/M)
        mu      = np.arcsin(1.0 / self.M1)
        beta_lo = mu + 1e-5
        beta_hi = np.radians(89.9)

        # Scan for sign change (weak-shock solution)
        betas = np.linspace(beta_lo, np.radians(80.0), 10_000)
        vals  = np.array([self._tbm_residual(b) for b in betas])
        sign_changes = np.where(np.diff(np.sign(vals)))[0]

        if len(sign_changes) == 0:
            raise ValueError(
                f"No oblique shock solution found for M={self.M1}, "
                f"θ={np.degrees(self.theta):.1f}°. "
                f"Wedge angle may exceed the detachment limit."
            )
        # Take the first (weak shock) solution
        i = sign_changes[0]
        return brentq(self._tbm_residual, betas[i], betas[i + 1], xtol=1e-10)

    # ── Rankine-Hugoniot jump conditions ─────────────────────────────────

    def _rankine_hugoniot(self) -> dict:
        M1n = self.M1 * np.sin(self.beta)
        g   = self.gamma

        p2_p1   = 1.0 + 2.0 * g / (g + 1) * (M1n**2 - 1.0)
        rho2_r1 = (g + 1) * M1n**2 / ((g - 1) * M1n**2 + 2.0)
        T2_T1   = p2_p1 / rho2_r1

        # Post-shock Mach number
        M2n = np.sqrt(((g - 1) * M1n**2 + 2.0) / (2.0 * g * M1n**2 - (g - 1)))
        M2  = M2n / np.sin(self.beta - self.theta)

        return {
            "p":   self.p1   * p2_p1,
            "rho": self.rho1 * rho2_r1,
            "T":   self.T1   * T2_T1,
            "M":   M2,
        }

    # ── Velocity decomposition ────────────────────────────────────────────

    def _velocity(self, M: float, T: float, flow_angle: float):
        """Return (u, v) given Mach, temperature, and flow direction angle."""
        a   = np.sqrt(self.gamma * self.R * T)
        spd = M * a
        return spd * np.cos(flow_angle), spd * np.sin(flow_angle)

    # ── Point generation ──────────────────────────────────────────────────

    def generate(self) -> FlowData:
        """
        Scatter random collocation points in the domain above the wedge surface.
        Assign exact pre- or post-shock values based on shock geometry.
        """
        xmin, xmax, ymin, ymax = self.domain
        rng = np.random.default_rng(42)

        # Oversample and filter: keep only points above wedge y ≥ x·tan(θ)
        n_sample = self.n_pts * 5
        x_all = rng.uniform(xmin, xmax, n_sample)
        y_all = rng.uniform(ymin, ymax, n_sample)
        mask  = y_all >= x_all * np.tan(self.theta) + 1e-5
        x = x_all[mask][:self.n_pts]
        y = y_all[mask][:self.n_pts]

        if len(x) < self.n_pts:
            raise RuntimeError(
                f"Could not generate {self.n_pts} points above the wedge. "
                f"Try a larger domain or smaller wedge angle."
            )

        # Classify: above shock line y ≥ x·tan(β) → pre-shock (region 1)
        above_shock = y >= x * np.tan(self.beta)

        dn = self._post
        u1, v1 = self._velocity(self.M1,  self.T1,   0.0)         # horizontal
        u2, v2 = self._velocity(dn["M"],  dn["T"],  -self.theta)  # deflected down

        a1 = np.sqrt(self.gamma * self.R * self.T1)
        a2 = np.sqrt(self.gamma * self.R * dn["T"])

        rho = np.where(above_shock, self.rho1, dn["rho"]).astype(np.float32)
        u   = np.where(above_shock, u1,        u2       ).astype(np.float32)
        v   = np.where(above_shock, v1,        v2       ).astype(np.float32)
        p   = np.where(above_shock, self.p1,   dn["p"]  ).astype(np.float32)
        T   = np.where(above_shock, self.T1,   dn["T"]  ).astype(np.float32)
        a   = np.where(above_shock, a1,        a2       )
        Ma  = (np.sqrt(u**2 + v**2) / a).astype(np.float32)

        print(f"[WedgeSynth] Generated {len(x)} points "
              f"({above_shock.sum()} pre-shock, {(~above_shock).sum()} post-shock)")

        return FlowData(
            x=x.astype(np.float32), y=y.astype(np.float32),
            rho=rho, u=u, v=v, p=p, T=T, Ma=Ma,
        )

    @property
    def shock_angle_deg(self) -> float:
        return float(np.degrees(self.beta))

    @property
    def wedge_angle_deg(self) -> float:
        return float(np.degrees(self.theta))


# ── Data normaliser ──────────────────────────────────────────────────────────

class DataNormalizer:
    """
    Min-max normalises each FlowData field to the interval [-1, 1].

    Fitting is done on the training data; the same statistics are used
    to transform/inverse-transform all subsequent data (validation, test).

    This is critical for PINN training because:
    - Pressure (~100,000 Pa) and density (~1.2 kg/m³) differ by 5 orders of
      magnitude. Without normalisation the pressure gradient dominates and
      the network ignores density entirely.
    - The [-1, 1] range matches the tanh/SIREN output range, keeping
      gradients well-conditioned throughout training.
    """

    def __init__(self):
        self.stats: Dict[str, Tuple[float, float]] = {}
        self._fitted = False

    def fit(self, data: FlowData) -> "DataNormalizer":
        """Compute (min, max) statistics from training data."""
        for f in fields(data):
            arr = getattr(data, f.name)
            self.stats[f.name] = (float(arr.min()), float(arr.max()))
        self._fitted = True
        return self

    def transform(self, data: FlowData) -> FlowData:
        """Apply normalisation using fitted statistics."""
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform()")
        normed = {}
        for f in fields(data):
            arr     = getattr(data, f.name)
            lo, hi  = self.stats[f.name]
            span    = hi - lo if abs(hi - lo) > 1e-12 else 1.0
            normed[f.name] = ((2.0 * (arr - lo) / span) - 1.0).astype(np.float32)
        return FlowData(**normed)

    def inverse_transform_field(self, field: str, arr: np.ndarray) -> np.ndarray:
        """Denormalise a single field back to physical units."""
        lo, hi = self.stats[field]
        span   = hi - lo if abs(hi - lo) > 1e-12 else 1.0
        return ((arr + 1.0) / 2.0 * span + lo).astype(np.float32)

    def inverse_transform_tensor(self, field: str, t: torch.Tensor) -> torch.Tensor:
        """In-graph denormalisation (keeps grad_fn for PDE residuals)."""
        lo, hi = self.stats[field]
        span   = hi - lo if abs(hi - lo) > 1e-12 else 1.0
        return (t + 1.0) / 2.0 * span + lo

    def physical_coord_scale(self) -> Tuple[float, float]:
        """Return (span_x, span_y) for chain-rule correction in PDE derivatives."""
        sx = self.stats["x"][1] - self.stats["x"][0]
        sy = self.stats["y"][1] - self.stats["y"][0]
        return float(sx), float(sy)

    def save(self, path: str):
        """Persist statistics to a .npz file."""
        np.savez(path, **{k: np.array(v) for k, v in self.stats.items()})

    @classmethod
    def load(cls, path: str) -> "DataNormalizer":
        """Load statistics from a .npz file."""
        obj  = cls()
        data = np.load(path)
        obj.stats   = {k: (float(data[k][0]), float(data[k][1])) for k in data}
        obj._fitted = True
        return obj
