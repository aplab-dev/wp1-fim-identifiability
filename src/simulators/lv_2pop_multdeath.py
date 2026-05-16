"""Two-population Lotka-Volterra simulator with multiplicative drug-death entry.

Implements the model from `docs/notes/derivations.md` Derivation 1:

    dS/dt = r_S * S * (1 - (S + alpha * R) / K) - d * u(t) * S
    dR/dt = r_R * R * (1 - (R + beta * S) / K)

This is the *theory-tribe* parameterization (Strobl 2021 / Gallagher 2025 /
Wang & Lei 2025). Drug enters as a multiplicative death term on sensitive
cells, with control u(t) ∈ [0, 1].

Provides:
- :class:`LV2PopParams` — typed parameter container.
- :class:`LV2PopMultDeath` — simulator with ``dynamics``, ``simulate``,
  ``fixed_points``, ``stability`` methods.
- Closed-form fixed-point recovery and Jacobian-based stability classification
  per Derivation 1 §1.2-1.3.

Convention: state vector is ``x = [S, R]``. Time in days.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp


# ---------- parameter container ----------

@dataclass(frozen=True)
class LV2PopParams:
    """Parameters for the 2-population multiplicative-death L-V model.

    Defaults follow Strobl et al. 2021 / Gallagher 2025 (theory tribe).
    See ``docs/notes/biology.md`` for value ranges and references.
    """

    r_S: float = 0.027
    """Sensitive-cell intrinsic growth rate (per day). Typical range 0.02–0.05."""

    r_R: float = 0.020
    """Resistant-cell intrinsic growth rate (per day). Typical r_R ≤ r_S (resistance cost)."""

    alpha: float = 1.0
    """Effect of resistant on sensitive (competition coefficient). Typical 0.4–1.2."""

    beta: float = 1.0
    """Effect of sensitive on resistant. Typical 0.4–1.2."""

    K: float = 1.0
    """Shared carrying capacity. Often normalized to 1."""

    d: float = 1.5
    """Drug-induced death rate of sensitive cells at u=1. Strobl 2021 default."""

    def __post_init__(self) -> None:
        if self.r_S <= 0 or self.r_R < 0:
            raise ValueError(f"growth rates must be non-negative; r_S={self.r_S}, r_R={self.r_R}")
        if self.K <= 0:
            raise ValueError(f"carrying capacity must be positive; K={self.K}")
        if self.d < 0:
            raise ValueError(f"drug death rate must be non-negative; d={self.d}")


# ---------- simulation result ----------

@dataclass(frozen=True)
class LV2PopResult:
    """Output of a forward simulation."""

    t: np.ndarray
    """Time points (days)."""

    S: np.ndarray
    """Sensitive population trajectory."""

    R: np.ndarray
    """Resistant population trajectory."""

    u: np.ndarray
    """Drug control trajectory u(t) sampled at t."""

    params: LV2PopParams = field(repr=False)
    """Parameters used for the simulation."""

    @property
    def total(self) -> np.ndarray:
        """Total tumor burden S + R at each time point."""
        return self.S + self.R


# ---------- simulator ----------

class LV2PopMultDeath:
    """Two-population L-V with multiplicative drug-death.

    Stateless beyond the parameter container. Simulate with ``simulate()``;
    inspect equilibria with ``fixed_points()``; classify stability with
    ``stability()``.
    """

    def __init__(self, params: LV2PopParams | None = None) -> None:
        self.params = params or LV2PopParams()

    # --- core dynamics ---

    def dynamics(self, t: float, x: np.ndarray, u: float) -> np.ndarray:
        """Right-hand side of the L-V ODE at time t with state x and control u.

        Args:
            t: Current time (unused; kept for ``solve_ivp`` compatibility).
            x: State [S, R].
            u: Drug control in [0, 1].

        Returns:
            dx/dt as a length-2 array.
        """
        S, R = x[0], x[1]
        p = self.params
        dS = p.r_S * S * (1.0 - (S + p.alpha * R) / p.K) - p.d * u * S
        dR = p.r_R * R * (1.0 - (R + p.beta * S) / p.K)
        return np.array([dS, dR])

    def simulate(
        self,
        x0: tuple[float, float] | np.ndarray,
        t_span: tuple[float, float],
        control: Callable[[float], float] | float = 0.0,
        t_eval: np.ndarray | None = None,
        method: str = "RK45",
        rtol: float = 1e-6,
        atol: float = 1e-9,
        max_step: float | None = None,
    ) -> LV2PopResult:
        """Forward-simulate the system from x0 over t_span under given control.

        Args:
            x0: Initial state (S0, R0).
            t_span: (t_start, t_end) in days.
            control: Either a constant in [0, 1] or a callable u(t) returning a scalar.
            t_eval: Optional times at which to record state. Defaults to 200 points
                evenly spaced across t_span.
            method: ``solve_ivp`` integration method. RK45 is the project default.
            rtol, atol: Relative and absolute integration tolerances.
            max_step: Optional maximum step size. Useful when control(t) has
                discontinuities (drug switches).

        Returns:
            LV2PopResult with t, S, R, u arrays and the parameters used.

        Raises:
            ValueError: If x0 has wrong shape, t_span is invalid, or solver fails.
        """
        x0_arr = np.asarray(x0, dtype=float)
        if x0_arr.shape != (2,):
            raise ValueError(f"x0 must be length-2; got shape {x0_arr.shape}")
        if t_span[1] <= t_span[0]:
            raise ValueError(f"t_span must satisfy t1 > t0; got {t_span}")

        u_fn = control if callable(control) else (lambda _t: float(control))

        def rhs(t: float, x: np.ndarray) -> np.ndarray:
            return self.dynamics(t, x, u_fn(t))

        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 200)

        kwargs: dict = dict(
            t_span=t_span,
            y0=x0_arr,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )
        if max_step is not None:
            kwargs["max_step"] = max_step

        sol = solve_ivp(rhs, **kwargs)
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")

        u_arr = np.array([u_fn(t) for t in sol.t])
        return LV2PopResult(
            t=sol.t,
            S=sol.y[0],
            R=sol.y[1],
            u=u_arr,
            params=self.params,
        )

    # --- equilibrium analysis ---

    FixedPointKind = Literal["extinction", "S_only", "R_only", "coexistence"]

    def fixed_points(self, u: float = 0.0) -> dict[FixedPointKind, np.ndarray]:
        """Return the four candidate fixed points of the autonomous system at constant u.

        At u=0 (no drug), per Derivation 1 §1.2:
        - extinction: (0, 0)
        - S_only: (K, 0)
        - R_only: (0, K)
        - coexistence: (K(1-α)/(1-αβ), K(1-β)/(1-αβ)) if denominator non-zero

        At u>0, the S-only and coexistence fixed points shift; this method
        currently only returns the u=0 fixed points (the most useful for
        steady-state analysis). For non-zero u, use ``simulate`` and check
        long-time behavior.

        Args:
            u: Drug control level (only u=0 is fully supported analytically).

        Returns:
            Dict mapping fixed-point kind to its (S, R) location.

        Raises:
            NotImplementedError: For u != 0 (use simulate for that regime).
            ValueError: If the coexistence denominator vanishes (αβ = 1).
        """
        if u != 0.0:
            raise NotImplementedError(
                "Analytical fixed points implemented only for u=0. "
                "For non-zero u, simulate to long times and observe attractors."
            )
        p = self.params
        denom = 1.0 - p.alpha * p.beta
        if abs(denom) < 1e-12:
            raise ValueError(
                f"Coexistence fixed point ill-defined: αβ = {p.alpha * p.beta} ≈ 1. "
                "The system is at the bifurcation between weak and strong competition."
            )
        S_coex = p.K * (1.0 - p.alpha) / denom
        R_coex = p.K * (1.0 - p.beta) / denom
        return {
            "extinction": np.array([0.0, 0.0]),
            "S_only": np.array([p.K, 0.0]),
            "R_only": np.array([0.0, p.K]),
            "coexistence": np.array([S_coex, R_coex]),
        }

    def jacobian(self, x: np.ndarray, u: float = 0.0) -> np.ndarray:
        """Jacobian of the dynamics at state x and control u.

        From Derivation 1 §1.3:

            ∂f_S/∂S = r_S(1 - (2S + αR)/K) - d*u
            ∂f_S/∂R = -r_S * α * S / K
            ∂f_R/∂S = -r_R * β * R / K
            ∂f_R/∂R = r_R(1 - (2R + βS)/K)

        Args:
            x: State [S, R].
            u: Drug control level.

        Returns:
            2×2 Jacobian matrix.
        """
        S, R = x[0], x[1]
        p = self.params
        J = np.array(
            [
                [
                    p.r_S * (1.0 - (2.0 * S + p.alpha * R) / p.K) - p.d * u,
                    -p.r_S * p.alpha * S / p.K,
                ],
                [
                    -p.r_R * p.beta * R / p.K,
                    p.r_R * (1.0 - (2.0 * R + p.beta * S) / p.K),
                ],
            ]
        )
        return J

    StabilityKind = Literal["stable_node", "stable_focus", "saddle", "unstable_node", "unstable_focus", "center", "degenerate"]

    def stability(self, x: np.ndarray, u: float = 0.0) -> tuple[StabilityKind, np.ndarray]:
        """Classify the stability of a fixed point via Jacobian eigenvalues.

        Args:
            x: Fixed point (S*, R*).
            u: Drug control level used to evaluate the Jacobian.

        Returns:
            (label, eigenvalues) tuple. Label is one of the StabilityKind literals.
        """
        J = self.jacobian(x, u)
        eigvals = np.linalg.eigvals(J)
        re = eigvals.real
        im = eigvals.imag
        has_complex = np.any(np.abs(im) > 1e-9)

        if np.any(np.abs(re) < 1e-9):
            return "degenerate", eigvals
        all_negative = np.all(re < 0)
        all_positive = np.all(re > 0)

        if has_complex:
            if all_negative:
                return "stable_focus", eigvals
            if all_positive:
                return "unstable_focus", eigvals
            return "center", eigvals  # purely imaginary or mixed; rare in non-conservative systems
        # all real
        if all_negative:
            return "stable_node", eigvals
        if all_positive:
            return "unstable_node", eigvals
        return "saddle", eigvals
