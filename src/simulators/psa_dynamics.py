"""PSA dynamics — first-order linear ODE filter on cell counts.

Implements the PSA model from Zhang et al. 2017 (reassessment §"Correction 3"):

    dPSA/dt = sum_i x_i - phi * PSA

where each cell produces 1 PSA unit per day and PSA decays at rate phi.

Zhang 2017 uses phi = 0.5/day (half-life ~1.4 days). Brady-Nicholls 2020
uses phi = 0.0856/day (half-life ~8 days). We expose phi as a parameter
since the literature disagrees.

The PSA filter is a first-order linear filter over cell counts. It can be
attached on top of any cell-count simulator (2-pop or 3-pop). Two integration
modes:

1. **Coupled** — integrate PSA jointly with the cell counts via solve_ivp.
   Use this when control depends on PSA (e.g., AT50 protocol triggers on PSA
   threshold, not on raw cell count).

2. **Standalone** — given a cell-count trajectory ``total(t)``, solve the
   PSA ODE separately. Use when cell counts are already simulated and we
   want to derive PSA post-hoc (Zhang 2017 reassessment §"Correction 4").
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class PSAParams:
    """Parameters for the PSA filter.

    Defaults follow Zhang 2017 (phi = 0.5/day, production = 1.0 per cell per day).
    """

    phi: float = 0.5
    """PSA decay rate (per day). Zhang 2017: 0.5 (half-life ~1.4 d).
    Brady-Nicholls 2020: 0.0856 (half-life ~8 d). Choose per-paper."""

    rho: float = 1.0
    """PSA production per cell per day. Zhang 2017 assumes 1 unit; rescaling
    is observation-equivalent. Set <1 to model resistant cells producing
    less PSA per cell (gamma-correction in Derivation 4 §4.1)."""

    weights: np.ndarray | None = None
    """Optional per-population weights. If provided, PSA is rho * weights @ x
    instead of rho * sum(x). Useful for the gamma-correction
    (e.g., weights = (1, 1, 0.1) means T- contributes 10% as much as T+/TP)."""

    def __post_init__(self) -> None:
        if self.phi <= 0:
            raise ValueError(f"phi must be positive; got {self.phi}")
        if self.rho < 0:
            raise ValueError(f"rho must be non-negative; got {self.rho}")
        if self.weights is not None:
            if not isinstance(self.weights, np.ndarray):
                raise ValueError("weights must be ndarray if provided")
            if np.any(self.weights < 0):
                raise ValueError("weights must be non-negative")


def psa_steady_state(total_cells: float, params: PSAParams) -> float:
    """Quasi-steady-state PSA for a constant cell count.

    From dPSA/dt = rho * total - phi * PSA, setting dPSA/dt = 0:
        PSA_ss = rho * total / phi

    Useful sanity check and equilibrium value.

    Args:
        total_cells: Cell count (scalar or array; if weights are used,
            this should be the weighted sum already).
        params: PSAParams.

    Returns:
        Quasi-steady-state PSA value.
    """
    return params.rho * total_cells / params.phi


def integrate_psa_from_cells(
    t: np.ndarray,
    cells_by_t: np.ndarray | Callable[[float], np.ndarray],
    params: PSAParams | None = None,
    psa0: float | None = None,
) -> np.ndarray:
    """Integrate PSA dynamics given a cell-count trajectory.

    Standalone mode: cells are already simulated; we filter to get PSA.

    Args:
        t: Time grid for the output PSA array.
        cells_by_t: Either an (N_pop, N_t) array (cells aligned with t) or
            a callable returning the cell-count vector at any time.
        params: PSA parameters. Default = PSAParams() = Zhang 2017 settings.
        psa0: Initial PSA value. Default = quasi-steady-state at the
            initial cell count.

    Returns:
        PSA trajectory aligned with t.
    """
    p = params or PSAParams()

    if callable(cells_by_t):
        cell_fn = cells_by_t
    else:
        cells_arr = np.asarray(cells_by_t)
        if cells_arr.ndim != 2 or cells_arr.shape[1] != len(t):
            raise ValueError(
                f"cells_by_t array must be (N_pop, N_t={len(t)}); got {cells_arr.shape}"
            )

        # linear interpolation in time for solve_ivp's adaptive stepping
        def cell_fn(tau: float) -> np.ndarray:
            return np.array([
                np.interp(tau, t, cells_arr[i]) for i in range(cells_arr.shape[0])
            ])

    def total_cells(tau: float) -> float:
        x = cell_fn(tau)
        if p.weights is not None:
            if x.shape != p.weights.shape:
                raise ValueError(
                    f"weights shape {p.weights.shape} != cell vector shape {x.shape}"
                )
            return float(p.weights @ x)
        return float(np.sum(x))

    # initial PSA: quasi-steady-state at t[0]
    if psa0 is None:
        psa0 = psa_steady_state(total_cells(t[0]), p)

    def rhs(tau: float, psa: np.ndarray) -> np.ndarray:
        return np.array([p.rho * total_cells(tau) - p.phi * psa[0]])

    sol = solve_ivp(
        rhs,
        t_span=(t[0], t[-1]),
        y0=np.array([psa0]),
        t_eval=t,
        method="RK45",
        rtol=1e-6,
        atol=1e-6,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed for PSA filter: {sol.message}")
    return sol.y[0]
