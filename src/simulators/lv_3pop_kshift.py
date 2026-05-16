"""Three-population Lotka-Volterra simulator with K-shift drug entry.

Implements the canonical clinical mCRPC model from Zhang et al. 2017 and
Cunningham et al. 2020. The three cell types are:

- ``T+``  (testosterone-dependent; most androgen-axis-sensitive)
- ``TP``  (testosterone-producing autocrine)
- ``T-``  (testosterone-independent; resistant)

Drug enters via *carrying-capacity shift* — abiraterone collapses $K_{T+}$
and $K_{TP}$ to small values while leaving $K_{T-}$ intact. Per Zhang 2017
(reassessment §"Correction 2") and Cunningham 2020 (reassessment §"Correction 2"):

    K_{T-}(Λ)  = 10_000                   (drug-independent)
    K_{TP}(Λ)  = 10_000 - 9_900 * Λ       (linear drop, 100× under MTD)
    K_{T+}(Λ)  = (1.5 - Λ) * x_{TP}       (state-dependent symbiosis)

Dynamics (Cunningham 2020 Eq. 1):

    dx_i/dt = r_i * x_i * (K_i(Λ) - sum_j alpha_{ij} x_j) / K_i(Λ)

Growth rates: r = (2.7726, 3.4657, 6.6542) × 10⁻³ per day.

This is the *clinical-tribe* parameterization (companion to the
theory-tribe ``lv_2pop_multdeath`` simulator). See [docs/notes/field_map.md]
modeling-tribes section for the taxonomy.

Derivation references:
- Zhang 2017 reassessment in [docs/literature/zhang-2017-crpc-adaptive.md].
- Cunningham 2020 reassessment in [docs/literature/cunningham-2020-optimal-control.md].
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp


# ---------- parameter container ----------

@dataclass(frozen=True)
class LV3PopParams:
    """Parameters for the 3-population K-shift L-V model.

    Defaults match Cunningham et al. 2020 (which inherits Zhang 2017's
    parameterization). See docs/notes/biology.md for value ranges.
    """

    # --- intrinsic growth rates (per day; from Cunningham 2020 §"Correction 1") ---
    r_Tplus: float = 2.7726e-3
    """Growth rate of T+ cells (testosterone-dependent)."""
    r_TP: float = 3.4657e-3
    """Growth rate of TP cells (testosterone-producing)."""
    r_Tminus: float = 6.6542e-3
    """Growth rate of T- cells (testosterone-independent / resistant). Note: r_T- > r_T+."""

    # --- carrying-capacity parameters (from Zhang 2017 / Cunningham 2020) ---
    K_Tminus: float = 1e4
    """Carrying capacity of T- cells (drug-independent)."""
    K_TP_max: float = 1e4
    """Untreated carrying capacity of TP cells. Reduces under abiraterone."""
    K_TP_drop: float = 9_900.0
    """How much K_TP drops at full dose (Λ=1). K_TP(Λ) = K_TP_max - K_TP_drop * Λ."""
    mu_max: float = 1.5
    """Untreated symbiosis coefficient: K_T+ = mu(Λ) * x_TP, mu(0) = mu_max."""
    mu_drop: float = 1.0
    """How much mu drops at full dose. mu(Λ) = mu_max - mu_drop * Λ."""

    # --- competition matrix (asymmetric; Zhang 2017 uses rank-orderings) ---
    # Default values are illustrative midpoints in the 0.4-0.9 range that
    # Zhang 2017 supplementary Table 1 uses. Sub-classes can override.
    alpha: np.ndarray = field(default_factory=lambda: np.array([
        # T+   TP   T-
        [1.0, 0.5, 0.7],  # row 0: effects on T+
        [0.6, 1.0, 0.5],  # row 1: effects on TP
        [0.4, 0.3, 1.0],  # row 2: effects on T-
    ]))

    def __post_init__(self) -> None:
        if any(r < 0 for r in (self.r_Tplus, self.r_TP, self.r_Tminus)):
            raise ValueError("growth rates must be non-negative")
        if self.K_Tminus <= 0 or self.K_TP_max <= 0:
            raise ValueError("carrying capacities must be positive")
        if self.K_TP_drop > self.K_TP_max:
            raise ValueError("K_TP_drop must be <= K_TP_max (else K_TP(1) < 0)")
        if not isinstance(self.alpha, np.ndarray) or self.alpha.shape != (3, 3):
            raise ValueError(f"alpha must be 3x3 ndarray; got {type(self.alpha).__name__} shape {getattr(self.alpha, 'shape', None)}")
        if np.any(self.alpha < 0):
            raise ValueError("alpha matrix entries must be non-negative")

    def K(self, Lambda: float, x_TP: float) -> tuple[float, float, float]:
        """Carrying capacities at drug level Λ and current TP-cell count.

        Args:
            Lambda: Drug control level in [0, 1].
            x_TP: Current TP-cell count (needed for K_T+'s symbiosis).

        Returns:
            (K_T+, K_TP, K_T-) tuple.
        """
        K_TP_now = self.K_TP_max - self.K_TP_drop * Lambda
        # Guard against negative K_TP under unusual parameter overrides
        K_TP_now = max(K_TP_now, 1e-6)
        mu = self.mu_max - self.mu_drop * Lambda
        # K_T+ depends on x_TP via the symbiosis coefficient. If TP is depleted,
        # K_T+ collapses too — biologically realistic since T+ requires TP-produced
        # androgens to grow.
        K_Tplus_now = max(mu * x_TP, 1e-6)
        return K_Tplus_now, K_TP_now, self.K_Tminus


# ---------- simulation result ----------

@dataclass(frozen=True)
class LV3PopResult:
    """Output of a 3-pop forward simulation."""

    t: np.ndarray
    """Time points (days)."""
    x_Tplus: np.ndarray
    """T+ population trajectory."""
    x_TP: np.ndarray
    """TP population trajectory."""
    x_Tminus: np.ndarray
    """T- population trajectory."""
    Lambda: np.ndarray
    """Drug control trajectory Λ(t) sampled at t."""
    params: LV3PopParams = field(repr=False)

    @property
    def total(self) -> np.ndarray:
        """Total tumor burden across all three populations."""
        return self.x_Tplus + self.x_TP + self.x_Tminus

    @property
    def x(self) -> np.ndarray:
        """Stacked (3, N) array of populations for matrix operations."""
        return np.stack([self.x_Tplus, self.x_TP, self.x_Tminus])


# ---------- simulator ----------

class LV3PopKShift:
    """Three-population L-V with K-shift drug entry (Zhang 2017 / Cunningham 2020).

    Stateless beyond the parameter container.
    """

    def __init__(self, params: LV3PopParams | None = None) -> None:
        self.params = params or LV3PopParams()

    def dynamics(self, t: float, x: np.ndarray, Lambda: float) -> np.ndarray:
        """Right-hand side of the 3-pop K-shift ODE.

        Args:
            t: Current time (unused; kept for solve_ivp compatibility).
            x: State [x_T+, x_TP, x_T-].
            Lambda: Drug control level in [0, 1].

        Returns:
            dx/dt as length-3 array.
        """
        x_Tplus, x_TP, x_Tminus = x[0], x[1], x[2]
        p = self.params
        K_Tplus, K_TP, K_Tminus = p.K(Lambda, x_TP)

        # competition sum sum_j alpha_{ij} x_j for each i
        # alpha[i, j] = effect of population j on population i
        # so the term for population i is alpha[i, :] @ x
        comp = p.alpha @ x

        # Per Cunningham 2020 Eq. 1: dx_i/dt = r_i * x_i * (K_i - sum_j alpha_ij x_j) / K_i
        dx_Tplus = p.r_Tplus * x_Tplus * (K_Tplus - comp[0]) / K_Tplus
        dx_TP = p.r_TP * x_TP * (K_TP - comp[1]) / K_TP
        dx_Tminus = p.r_Tminus * x_Tminus * (K_Tminus - comp[2]) / K_Tminus

        return np.array([dx_Tplus, dx_TP, dx_Tminus])

    def simulate(
        self,
        x0: tuple[float, float, float] | np.ndarray,
        t_span: tuple[float, float],
        control: Callable[[float], float] | float = 0.0,
        t_eval: np.ndarray | None = None,
        method: str = "LSODA",
        rtol: float = 1e-6,
        atol: float = 1e-6,
        max_step: float | None = None,
    ) -> LV3PopResult:
        """Forward-simulate from x0 over t_span.

        Args:
            x0: Initial state (x_T+, x_TP, x_T-).
            t_span: (t_start, t_end) in days.
            control: Constant in [0, 1] or callable Λ(t).
            t_eval: Optional sample times. Defaults to 200 points across t_span.
            method: solve_ivp method. **LSODA default** for the 3-pop K-shift
                model, NOT RK45 — under MTD, K_T+ collapses toward 0 (because
                K_T+ = mu * x_TP and x_TP collapses), making the dynamics stiff.
                RK45 is ~500× slower than LSODA in that regime. LSODA
                auto-detects stiffness and switches to BDF as needed.
            rtol, atol: Integration tolerances. Looser atol than 2-pop because
                3-pop K-shift involves much larger absolute populations (~10^4).
            max_step: Optional max step (useful at drug switches).

        Returns:
            LV3PopResult.

        Raises:
            ValueError: If x0 has wrong shape, t_span invalid.
            RuntimeError: If solve_ivp fails.
        """
        x0_arr = np.asarray(x0, dtype=float)
        if x0_arr.shape != (3,):
            raise ValueError(f"x0 must be length-3; got shape {x0_arr.shape}")
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

        Lambda_arr = np.array([u_fn(t) for t in sol.t])
        return LV3PopResult(
            t=sol.t,
            x_Tplus=sol.y[0],
            x_TP=sol.y[1],
            x_Tminus=sol.y[2],
            Lambda=Lambda_arr,
            params=self.params,
        )

    # ---------- equilibrium analysis ----------

    def equilibrium_under_no_drug(
        self,
        x0: tuple[float, float, float] | None = None,
        t_max: float = 5_000.0,
        atol: float = 5e-2,
    ) -> np.ndarray:
        """Find a coexistence equilibrium under Λ=0 by long-time simulation.

        Approach: simulate forward from a reasonable initial guess for a long
        time, then return the asymptotic state. More robust than naive
        fixed-point iteration — handles parameter regimes where the iteration
        wouldn't contract.

        **Slow-approach caveat.** The 3-pop K-shift system with default
        parameters has multi-time-scale dynamics: $TP$ and $T-$ equilibrate
        in ~1000-3000 days, but $T+$ drifts very slowly (its growth rate is
        $r_{T+} = 2.7726 \\times 10^{-3}$/day and its target depends on $TP$
        through symbiosis). The *true* equilibrium is approached only at
        very long times (~10^5 days for default parameters).

        For Phase 2 reproduction purposes (Zhang 2017's "ESS at 25% of
        untreated equilibrium"), the t_max=5000 default returns a state
        close enough to the asymptotic equilibrium for the fast components
        (TP, T-) but with T+ still slowly drifting. This is biologically
        appropriate — patients live on the order of years, not centuries.
        Tests should use loose tolerances (~10-15% rel) on T+ between
        different initial conditions.

        Args:
            x0: Optional initial guess. Default: (K_Tminus/4, K_TP_max/2,
                K_Tminus/2), seeded with non-trivial counts in all three
                populations so none goes extinct due to numerical floor.
            t_max: Simulation horizon for long-time approximation.
            atol: Tolerance on the rate-of-change norm at the final state
                (verifies we're actually at quasi-equilibrium).

        Returns:
            Equilibrium state x* = (x_T+, x_TP, x_T-).

        Raises:
            RuntimeError: If the final state is not at quasi-equilibrium
                (|dx/dt| > atol component-wise).
        """
        p = self.params
        if x0 is None:
            x0 = (p.K_Tminus / 4, p.K_TP_max / 2, p.K_Tminus / 2)
        result = self.simulate(
            x0=x0,
            t_span=(0.0, t_max),
            control=0.0,
            t_eval=np.array([t_max]),  # only need the endpoint
        )
        x_final = np.array([result.x_Tplus[-1], result.x_TP[-1], result.x_Tminus[-1]])
        # Verify quasi-equilibrium
        dx_final = self.dynamics(t_max, x_final, Lambda=0.0)
        rel_dx = np.abs(dx_final) / np.maximum(np.abs(x_final), 1.0)
        if np.any(rel_dx > atol):
            raise RuntimeError(
                f"State at t_max={t_max} not at quasi-equilibrium: "
                f"x={x_final}, dx={dx_final}, rel_dx={rel_dx}, atol={atol}. "
                f"Try increasing t_max or check if a stable equilibrium exists."
            )
        return x_final
