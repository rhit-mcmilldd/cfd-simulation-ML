"""
data/loader.py
----------------
Handles all data ingestion for the Shockwave PINN project:

  1. SU2Loader — reads SU2 CSV surface or solution output.
  2. WedgeSyntheticData — generates exact oblique-shock data from the
                          Rankine-Hugoniot jump conditions (no CFD needed).
  3. DataNormalizer — min-max scales all fields to [-1, 1] for stable training.
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


def _is_float_string(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _normalize_header(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


class SU2Loader:
    """
    Reads SU2 CSV surface/solution data produced by SU2.

    Accepts a CSV file path or a directory containing a single CSV file.
    Supports headers with names such as x, y, rho, u, v, p, T, Ma.
    If the Mach column is missing, Mach is computed from U and T.
    """

    DEFAULT_FILENAMES = [
        "surface.csv",
        "solution.csv",
        "su2_surface.csv",
        "flowfield.csv",
        "solution_surface.csv",
    ]

    def __init__(self, data_path: str):
        self.root = Path(data_path)
        if self.root.is_dir():
            self.filepath = self._find_csv_in_dir(self.root)
        else:
            self.filepath = self.root

        if not self.filepath.exists():
            raise FileNotFoundError(
                f"SU2 data file not found: {self.filepath}\n"
                f"Provide a CSV file or directory containing SU2 CSV output."
            )

    def _find_csv_in_dir(self, root: Path) -> Path:
        candidates = [root / name for name in self.DEFAULT_FILENAMES if (root / name).exists()]
        if len(candidates) == 1:
            return candidates[0]

        csv_files = sorted(root.glob("*.csv"))
        if len(csv_files) == 1:
            return csv_files[0]

        if candidates:
            return candidates[0]

        raise FileNotFoundError(
            f"No supported SU2 CSV file found in {root}. "
            "Expected one of: {} or any single .csv file.".format(
                ", ".join(self.DEFAULT_FILENAMES)
            )
        )

    def _read_csv(self, filepath: Path) -> tuple[np.ndarray, list[str] | None]:
        lines = []
        with filepath.open("r", encoding="utf-8", newline="") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or line.startswith("%"):
                    continue
                lines.append(line)

        if not lines:
            raise RuntimeError(f"No CSV data parsed from {filepath}")

        first = [token.strip() for token in lines[0].split(",")]
        if any(not _is_float_string(token) for token in first):
            header = [_normalize_header(token) for token in first]
            data_lines = lines[1:]
        else:
            header = None
            data_lines = lines

        data = np.genfromtxt(data_lines, delimiter=",", dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        return data, header

    def _find_column(self, header: list[str] | None, data: np.ndarray, *names: str) -> np.ndarray:
        if header is not None:
            for name in names:
                normalized = _normalize_header(name)
                if normalized in header:
                    return data[:, header.index(normalized)]
            raise ValueError(f"SU2 CSV file is missing required column {names[0]}.")

        if data.shape[1] < 8:
            raise ValueError(
                f"Unsupported SU2 CSV shape: {data.shape}. Expected at least 8 columns."
            )

        mapping = {
            "x": 0,
            "y": 1,
            "z": 2,
            "rho": 3,
            "u": 4,
            "v": 5,
            "p": 6,
            "t": 7,
            "ma": 8,
        }
        for name in names:
            key = _normalize_header(name)
            if key in mapping and mapping[key] < data.shape[1]:
                return data[:, mapping[key]]

        raise ValueError(f"Unable to map SU2 CSV column names {names}.")

    def load(self) -> FlowData:
        data, header = self._read_csv(self.filepath)

        x = self._find_column(header, data, "x")
        y = self._find_column(header, data, "y")
        rho = self._find_column(header, data, "rho", "density")
        u = self._find_column(header, data, "u", "u_velocity", "ux")
        v = self._find_column(header, data, "v", "v_velocity", "uy")
        p = self._find_column(header, data, "p", "pressure")
        T = self._find_column(header, data, "t", "temperature")

        if header is not None and any(name in header for name in ("ma", "mach")):
            Ma = self._find_column(header, data, "ma", "mach")
        else:
            a = np.sqrt(GAMMA * R_AIR * np.maximum(T, 1.0))
            Ma = np.sqrt(u ** 2 + v ** 2) / a
            print("[SU2Loader] Mach field not found in CSV — computed from U and T.")

        x = x.astype(np.float32)
        y = y.astype(np.float32)
        rho = rho.astype(np.float32)
        u = u.astype(np.float32)
        v = v.astype(np.float32)
        p = p.astype(np.float32)
        T = T.astype(np.float32)
        Ma = Ma.astype(np.float32)

        n = len(x)
        print(f"[SU2Loader] Loaded {n} surface points from {self.filepath}")
        data_obj = FlowData(x=x, y=y, rho=rho, u=u, v=v, p=p, T=T, Ma=Ma)
        data_obj.summary()
        return data_obj


# ── Synthetic data generator ─────────────────────────────────────────────────

class WedgeSyntheticData:
    """
    Generates an exact oblique-shock flow field using the Theta-Beta-Mach
    relation and Rankine-Hugoniot jump conditions.

    This is used for:
      - Prototyping the PINN before SU2 reference data is available
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
        seed:                 Optional[int] = None,
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
        self.seed   = seed

        self.beta = self._solve_shock_angle()
        self._post = self._rankine_hugoniot()

        print(f"[WedgeSynth] M_inf={self.M1}  theta={wedge_half_angle_deg:.1f} deg  "
              f"beta={np.degrees(self.beta):.2f} deg")
        print(f"[WedgeSynth] Post-shock: M={self._post['M']:.3f}  "
              f"p={self._post['p']:.0f} Pa  T={self._post['T']:.1f} K  "
              f"rho={self._post['rho']:.4f} kg/m3")

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
        rng = np.random.default_rng(self.seed)

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
