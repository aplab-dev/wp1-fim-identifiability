"""Tests for the 2-population L-V multiplicative-death simulator.

Verifies the core analytic predictions from `docs/notes/derivations.md` Derivation 1:
- Four fixed points at u=0 (extinction, S-only, R-only, coexistence).
- Stability classification matches the four regimes (A/B/C/D).
- Long-time simulation converges to the predicted attractor.
- Treated case (u=1, d > r_S) drives S → 0 monotonically.

Conventions per `tests/README.md`:
- Numerical tests use ``numpy.testing.assert_allclose`` with explicit tolerances.
- Each test is independent; ``pytest.fixture`` for setup; no module-level state.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulators.lv_2pop_multdeath import LV2PopMultDeath, LV2PopParams


# ---------- fixtures ----------

@pytest.fixture
def regime_A_sim() -> LV2PopMultDeath:
    """Regime A (weak competition, αβ < 1): coexistence stable.

    Per Derivation 1 §1.4, this regime has a stable interior fixed point.
    """
    p = LV2PopParams(r_S=0.05, r_R=0.04, alpha=0.7, beta=0.6, K=1.0, d=1.5)
    return LV2PopMultDeath(p)


@pytest.fixture
def regime_C_sim() -> LV2PopMultDeath:
    """Regime C (α<1, β>1): S dominates. S-only fixed point stable."""
    p = LV2PopParams(r_S=0.05, r_R=0.04, alpha=0.5, beta=1.5, K=1.0, d=1.5)
    return LV2PopMultDeath(p)


@pytest.fixture
def regime_D_sim() -> LV2PopMultDeath:
    """Regime D (α>1, β<1): R dominates. R-only fixed point stable."""
    p = LV2PopParams(r_S=0.05, r_R=0.04, alpha=1.5, beta=0.5, K=1.0, d=1.5)
    return LV2PopMultDeath(p)


# ---------- parameter validation ----------

class TestParameterValidation:
    """Parameter container rejects invalid inputs."""

    def test_negative_growth_rate_rejected(self) -> None:
        with pytest.raises(ValueError, match="growth rates"):
            LV2PopParams(r_S=-0.1)

    def test_zero_growth_rate_rejected(self) -> None:
        with pytest.raises(ValueError, match="growth rates"):
            LV2PopParams(r_S=0.0)

    def test_negative_K_rejected(self) -> None:
        with pytest.raises(ValueError, match="carrying capacity"):
            LV2PopParams(K=-1.0)

    def test_negative_d_rejected(self) -> None:
        with pytest.raises(ValueError, match="drug death rate"):
            LV2PopParams(d=-0.5)

    def test_default_params_valid(self) -> None:
        p = LV2PopParams()
        assert p.r_S > 0 and p.K > 0


# ---------- fixed-point recovery ----------

class TestFixedPoints:
    """Closed-form fixed points match Derivation 1 §1.2."""

    def test_extinction_at_origin(self, regime_A_sim: LV2PopMultDeath) -> None:
        fps = regime_A_sim.fixed_points()
        np.testing.assert_allclose(fps["extinction"], [0.0, 0.0])

    def test_S_only_at_K(self, regime_A_sim: LV2PopMultDeath) -> None:
        fps = regime_A_sim.fixed_points()
        np.testing.assert_allclose(fps["S_only"], [regime_A_sim.params.K, 0.0])

    def test_R_only_at_K(self, regime_A_sim: LV2PopMultDeath) -> None:
        fps = regime_A_sim.fixed_points()
        np.testing.assert_allclose(fps["R_only"], [0.0, regime_A_sim.params.K])

    def test_coexistence_formula_matches_derivation(self) -> None:
        """Coexistence fixed point: S* = K(1-α)/(1-αβ), R* = K(1-β)/(1-αβ)."""
        p = LV2PopParams(alpha=0.5, beta=0.6, K=2.0)
        sim = LV2PopMultDeath(p)
        fps = sim.fixed_points()
        denom = 1.0 - p.alpha * p.beta  # = 0.7
        expected_S = p.K * (1.0 - p.alpha) / denom  # 2 * 0.5 / 0.7
        expected_R = p.K * (1.0 - p.beta) / denom  # 2 * 0.4 / 0.7
        np.testing.assert_allclose(fps["coexistence"], [expected_S, expected_R], rtol=1e-12)

    def test_fixed_point_at_zero_drug_is_actually_a_zero(self, regime_A_sim: LV2PopMultDeath) -> None:
        """Verify that the analytic FPs satisfy dynamics(x*) = 0 at u=0."""
        fps = regime_A_sim.fixed_points()
        for name, x in fps.items():
            dx = regime_A_sim.dynamics(0.0, x, u=0.0)
            np.testing.assert_allclose(
                dx, [0.0, 0.0], atol=1e-12,
                err_msg=f"Fixed point '{name}' at {x} has non-zero dynamics {dx}",
            )

    def test_coexistence_in_positive_quadrant_for_regime_A(self, regime_A_sim: LV2PopMultDeath) -> None:
        """Regime A (αβ<1, both <1) has coexistence FP in positive quadrant."""
        fps = regime_A_sim.fixed_points()
        S, R = fps["coexistence"]
        assert S > 0 and R > 0

    def test_bifurcation_singularity_raises(self) -> None:
        """At αβ = 1 (the bifurcation), coexistence formula is singular."""
        p = LV2PopParams(alpha=1.0, beta=1.0)  # αβ = 1
        sim = LV2PopMultDeath(p)
        with pytest.raises(ValueError, match="bifurcation"):
            sim.fixed_points()

    def test_nonzero_u_not_implemented(self, regime_A_sim: LV2PopMultDeath) -> None:
        """Analytic FPs only at u=0 currently."""
        with pytest.raises(NotImplementedError):
            regime_A_sim.fixed_points(u=0.5)


# ---------- stability classification ----------

class TestStability:
    """Stability matches Derivation 1 §1.4 phase-portrait table."""

    def test_extinction_unstable_for_regime_A(self, regime_A_sim: LV2PopMultDeath) -> None:
        """Origin is always an unstable node — a tumor never extinguishes spontaneously."""
        fps = regime_A_sim.fixed_points()
        kind, _ = regime_A_sim.stability(fps["extinction"])
        assert kind == "unstable_node"

    def test_S_only_saddle_in_regime_A(self, regime_A_sim: LV2PopMultDeath) -> None:
        """Regime A: β<1 makes S-only a saddle (R can invade)."""
        fps = regime_A_sim.fixed_points()
        kind, _ = regime_A_sim.stability(fps["S_only"])
        assert kind == "saddle"

    def test_S_only_stable_in_regime_C(self, regime_C_sim: LV2PopMultDeath) -> None:
        """Regime C: β>1 makes S-only stable (R cannot invade)."""
        fps = regime_C_sim.fixed_points()
        kind, _ = regime_C_sim.stability(fps["S_only"])
        assert kind == "stable_node"

    def test_R_only_saddle_in_regime_A(self, regime_A_sim: LV2PopMultDeath) -> None:
        """Regime A: α<1 makes R-only a saddle (S can invade)."""
        fps = regime_A_sim.fixed_points()
        kind, _ = regime_A_sim.stability(fps["R_only"])
        assert kind == "saddle"

    def test_R_only_stable_in_regime_D(self, regime_D_sim: LV2PopMultDeath) -> None:
        """Regime D: α>1 makes R-only stable (S cannot invade)."""
        fps = regime_D_sim.fixed_points()
        kind, _ = regime_D_sim.stability(fps["R_only"])
        assert kind == "stable_node"

    def test_coexistence_stable_in_regime_A(self, regime_A_sim: LV2PopMultDeath) -> None:
        """Regime A (weak competition αβ<1): coexistence is stable.

        Per Derivation 1 §1.3: det J > 0, tr J < 0 ⇒ stable node or focus.
        """
        fps = regime_A_sim.fixed_points()
        kind, _ = regime_A_sim.stability(fps["coexistence"])
        assert kind in ("stable_node", "stable_focus")


# ---------- forward simulation ----------

class TestSimulationForward:
    """Forward integration produces correct asymptotic behavior."""

    def test_simulation_basic_shape(self, regime_A_sim: LV2PopMultDeath) -> None:
        """simulate() returns arrays of consistent length."""
        result = regime_A_sim.simulate(x0=(0.5, 0.1), t_span=(0.0, 100.0), control=0.0)
        assert len(result.t) == len(result.S) == len(result.R) == len(result.u)
        assert result.t[0] == pytest.approx(0.0)
        assert result.t[-1] == pytest.approx(100.0)

    def test_no_drug_converges_to_coexistence_in_regime_A(
        self,
        regime_A_sim: LV2PopMultDeath,
    ) -> None:
        """Long-time simulation under u=0 reaches the coexistence FP in regime A."""
        result = regime_A_sim.simulate(
            x0=(0.5, 0.1),
            t_span=(0.0, 5_000.0),  # long enough to converge
            control=0.0,
        )
        fps = regime_A_sim.fixed_points()
        S_coex, R_coex = fps["coexistence"]
        np.testing.assert_allclose(result.S[-1], S_coex, rtol=1e-3)
        np.testing.assert_allclose(result.R[-1], R_coex, rtol=1e-3)

    def test_full_drug_drives_S_to_zero_in_regime_A(
        self,
        regime_A_sim: LV2PopMultDeath,
    ) -> None:
        """Under u=1 with d > r_S, sensitive cells decay to zero (Derivation 1 §1.5)."""
        # d = 1.5, r_S = 0.05, so d > r_S by a wide margin.
        result = regime_A_sim.simulate(
            x0=(0.5, 0.01),
            t_span=(0.0, 1_000.0),
            control=1.0,
        )
        # S should be effectively zero
        assert result.S[-1] < 1e-6
        # R should approach K (its own carrying capacity, no competition)
        np.testing.assert_allclose(result.R[-1], regime_A_sim.params.K, rtol=1e-2)

    def test_simulation_with_callable_control(
        self,
        regime_A_sim: LV2PopMultDeath,
    ) -> None:
        """Callable control u(t) is supported and applied correctly."""
        # Step function: drug off for t < 50, on after
        def step_control(t: float) -> float:
            return 0.0 if t < 50.0 else 1.0

        result = regime_A_sim.simulate(
            x0=(0.5, 0.05),
            t_span=(0.0, 200.0),
            control=step_control,
            max_step=1.0,  # tighten step around the discontinuity
        )
        # u trajectory recorded at sample times: 0 for t<50, 1 for t≥50
        assert np.all(result.u[result.t < 50.0] == 0.0)
        assert np.all(result.u[result.t >= 50.0] == 1.0)

    def test_invalid_x0_shape_rejected(self, regime_A_sim: LV2PopMultDeath) -> None:
        with pytest.raises(ValueError, match="length-2"):
            regime_A_sim.simulate(x0=(1.0, 2.0, 3.0), t_span=(0.0, 10.0))

    def test_invalid_t_span_rejected(self, regime_A_sim: LV2PopMultDeath) -> None:
        with pytest.raises(ValueError, match="t1 > t0"):
            regime_A_sim.simulate(x0=(0.5, 0.1), t_span=(10.0, 5.0))


# ---------- AT mechanism check ----------

class TestATMechanism:
    """Sanity-check the AT mechanism formula from Derivation 1 §1.6.

    Quote: "maintaining S > 0 shifts the resistant carrying capacity from K
    down to K - β·S."
    """

    def test_resistant_quasi_steady_state_matches_K_minus_betaS(self) -> None:
        """With S held approximately fixed at S*, the R-only ODE settles at K - βS*.

        We use a regime where coexistence is stable; the R-component of the
        coexistence fixed point should equal K - β·S*_{coex} (a logistic-only
        prediction). Check that the formula aligns up to the L-V cross-term
        correction.
        """
        p = LV2PopParams(r_S=0.05, r_R=0.04, alpha=0.5, beta=0.5, K=1.0)
        sim = LV2PopMultDeath(p)
        fps = sim.fixed_points()
        S_coex, R_coex = fps["coexistence"]
        # At coexistence, dR/dt = 0 ⇒ 1 - (R + β·S)/K = 0 (assuming R > 0)
        # ⇒ R = K - β·S
        np.testing.assert_allclose(R_coex, p.K - p.beta * S_coex, rtol=1e-12)
