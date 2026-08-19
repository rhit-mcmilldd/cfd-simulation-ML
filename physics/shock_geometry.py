"""
physics/shock_geometry.py
--------------------------
Standalone Theta-Beta-Mach solver.

The oblique shock angle beta is fully determined by the free-stream Mach
number and wedge half-angle via classical shock theory — it doesn't matter
whether the flow field itself came from the synthetic Rankine-Hugoniot
generator or a real SU2 CFD run. This lets both code paths locate the
analytic shock line (for physics-loss down-weighting near the
discontinuity, and for plotting) without needing to re-instantiate the
full WedgeSyntheticData generator.
"""

import numpy as np
from scipy.optimize import brentq


def solve_beta_rad(mach_inf: float, theta_rad: float, gamma: float = 1.4) -> float:
    """
    Solve the Theta-Beta-Mach relation for the weak-shock solution.

    Returns the shock angle beta in radians. Raises ValueError if no
    attached oblique shock solution exists for this Mach/wedge-angle
    combination (i.e. the wedge angle exceeds the detachment limit).
    """
    def tbm_residual(beta):
        num = mach_inf**2 * np.sin(beta)**2 - 1
        den = mach_inf**2 * (gamma + np.cos(2 * beta)) + 2
        return np.tan(theta_rad) - (2 / np.tan(beta)) * (num / den)

    mu = np.arcsin(1.0 / mach_inf)
    betas = np.linspace(mu + 1e-5, np.radians(80.0), 10_000)
    vals = np.array([tbm_residual(b) for b in betas])
    sign_changes = np.where(np.diff(np.sign(vals)))[0]

    if len(sign_changes) == 0:
        raise ValueError(
            f"No oblique shock solution for M={mach_inf}, "
            f"theta={np.degrees(theta_rad):.1f} deg — wedge angle likely "
            f"exceeds the detachment limit at this Mach number."
        )
    i = sign_changes[0]
    return brentq(tbm_residual, betas[i], betas[i + 1], xtol=1e-10)
