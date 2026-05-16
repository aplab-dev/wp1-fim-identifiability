"""Tests for the 3-population K-shift simulator.

Verifies:
- Parameter validation matches the constraints from Zhang 2017 / Cunningham 2020.
- K(Λ) carrying-capacity formulae match the published values at Λ=0 and Λ=1.
- The MTD limit (Λ=1) drives T+ and TP toward zero, T- toward K_T-.
- Equilibrium-finding by fixed-point iteration converges to a coexistence state
  in the untreated regime.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulators.lv_3pop_kshift import LV3PopKShift, LV3PopParams


# ---------- fixtures ----------

@pytest.fixture
def default_sim() -> LV3PopKShift:
    """Default parameters from Cunningham 2020."""
    return LV3PopKShift(LV3PopParams())


# ---------- parameter validation ----------

class TestParameterValidation:
    """LV3PopParams rejects invalid inputs."""

    def test_negative_growth_rate_rejected(self) -> None:
        with pytest.raises(ValueError, match="growth rates"):
            LV3PopParams(r_Tplus=-1e-3)

    def test_zero_K_Tminus_rejected(self) -> None:
        with pytest.raises(ValueError, match="carrying capacities"):
            LV3PopParams(K_Tminus=0.0)

    def test_K_TP_drop_too_large_rejected(self) -> None:
        with pytest.raises(ValueError, match="K_TP_drop"):
            LV3PopParams(K_TP_max=1000, K_TP_drop=2000)

    def test_negative_alpha_rejected(self) -> None:
        bad = np.array([
            [1.0, -0.1, 0.7],
            [0.6, 1.0, 0.5],
            [0.4, 0.3, 1.0],
        ])
        with pytest.raises(ValueError, match="alpha"):
            LV3PopParams(alpha=bad)

    def test_wrong_alpha_shape_rejected(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            LV3PopParams(alpha=np.eye(2))

    def test_default_params_valid(self) -> None:
        p = LV3PopParams()
        assert p.r_Tplus > 0
        assert p.alpha.shape == (3, 3)


# ---------- carrying-capacity formula ----------

class TestKShiftFormula:
    """K(Λ, x_TP) matches Zhang 2017 / Cunningham 2020 published values."""

    def test_K_at_no_drug(self, default_sim: LV3PopKShift) -> None:
        """Λ=0: K_TP = 10000, K_T- = 10000, K_T+ = 1.5 * x_TP."""
        x_TP = 5000.0
        K_Tplus, K_TP, K_Tminus = default_sim.params.K(0.0, x_TP)
        assert K_TP == pytest.approx(1e4)
        assert K_Tminus == pytest.approx(1e4)
        assert K_Tplus == pytest.approx(1.5 * x_TP)

    def test_K_at_full_drug(self, default_sim: LV3PopKShift) -> None:
        """Λ=1: K_TP -> 100 (10000 - 9900), K_T+ = 0.5 * x_TP, K_T- = 10000."""
        x_TP = 5000.0
        K_Tplus, K_TP, K_Tminus = default_sim.params.K(1.0, x_TP)
        assert K_TP == pytest.approx(100.0)
        assert K_Tminus == pytest.approx(1e4)
        assert K_Tplus == pytest.approx(0.5 * x_TP)

    def test_K_T_minus_drug_independent(self, default_sim: LV3PopKShift) -> None:
        """K_T- should not depend on Λ at all."""
        for Lambda in [0.0, 0.3, 0.7, 1.0]:
            _, _, K_Tminus = default_sim.params.K(Lambda, x_TP=1000.0)
            assert K_Tminus == pytest.approx(1e4)

    def test_K_TP_linear_in_Lambda(self, default_sim: LV3PopKShift) -> None:
        """K_TP(Λ) = K_TP_max - K_TP_drop * Λ; check linearity."""
        p = default_sim.params
        for Lambda in np.linspace(0.0, 1.0, 5):
            _, K_TP, _ = p.K(Lambda, x_TP=1000.0)
            expected = p.K_TP_max - p.K_TP_drop * Lambda
            assert K_TP == pytest.approx(expected)


# ---------- forward simulation ----------

class TestSimulationForward:
    """Forward integration produces sensible behavior."""

    def test_simulation_basic_shape(self, default_sim: LV3PopKShift) -> None:
        result = default_sim.simulate(
            x0=(1000.0, 5000.0, 100.0),
            t_span=(0.0, 100.0),
        )
        n = len(result.t)
        assert len(result.x_Tplus) == n
        assert len(result.x_TP) == n
        assert len(result.x_Tminus) == n
        assert len(result.Lambda) == n

    def test_total_property(self, default_sim: LV3PopKShift) -> None:
        """LV3PopResult.total returns sum of all three populations."""
        result = default_sim.simulate(x0=(100.0, 500.0, 10.0), t_span=(0.0, 50.0))
        np.testing.assert_allclose(
            result.total, result.x_Tplus + result.x_TP + result.x_Tminus
        )

    def test_x_property_returns_stacked_array(self, default_sim: LV3PopKShift) -> None:
        result = default_sim.simulate(x0=(100.0, 500.0, 10.0), t_span=(0.0, 50.0))
        assert result.x.shape == (3, len(result.t))

    def test_full_drug_drives_T_plus_TP_to_low(self, default_sim: LV3PopKShift) -> None:
        """Under Λ=1 with sustained MTD, TP and T+ collapse and T- expands.

        Per Zhang 2017 reassessment §"Correction 2": K_TP collapses 100x,
        and K_T+ collapses with TP via the symbiosis term. Long-time
        behavior should have x_T- dominant.
        """
        result = default_sim.simulate(
            x0=(1000.0, 5000.0, 100.0),
            t_span=(0.0, 5000.0),  # very long
            control=1.0,
        )
        # At end, T- should dominate
        end_total = result.total[-1]
        assert result.x_Tminus[-1] / end_total > 0.5, (
            f"T- fraction = {result.x_Tminus[-1] / end_total:.3f}, expected > 0.5"
        )
        # T+ should be small (its K collapsed because TP collapsed)
        assert result.x_Tplus[-1] / end_total < 0.3

    def test_no_drug_preserves_all_three_populations(self, default_sim: LV3PopKShift) -> None:
        """Under Λ=0 with all populations seeded, none should go extinct.

        With the default symmetric-ish alpha and well-balanced ICs, the
        no-drug regime should sustain all three populations at non-trivial
        levels (coexistence).
        """
        result = default_sim.simulate(
            x0=(2000.0, 5000.0, 1000.0),
            t_span=(0.0, 5000.0),
            control=0.0,
        )
        for arr in (result.x_Tplus, result.x_TP, result.x_Tminus):
            assert arr[-1] > 1.0, "population unexpectedly went extinct under Λ=0"

    def test_invalid_x0_shape_rejected(self, default_sim: LV3PopKShift) -> None:
        with pytest.raises(ValueError, match="length-3"):
            default_sim.simulate(x0=(1.0, 2.0), t_span=(0.0, 10.0))

    def test_invalid_t_span_rejected(self, default_sim: LV3PopKShift) -> None:
        with pytest.raises(ValueError, match="t1 > t0"):
            default_sim.simulate(x0=(100.0, 500.0, 10.0), t_span=(10.0, 5.0))

    def test_callable_control_supported(self, default_sim: LV3PopKShift) -> None:
        """Callable Λ(t) is supported."""

        def step(t: float) -> float:
            return 0.0 if t < 100.0 else 1.0

        result = default_sim.simulate(
            x0=(1000.0, 5000.0, 100.0),
            t_span=(0.0, 200.0),
            control=step,
            max_step=1.0,
        )
        assert np.all(result.Lambda[result.t < 100.0] == 0.0)
        assert np.all(result.Lambda[result.t >= 100.0] == 1.0)


# ---------- equilibrium ----------

class TestEquilibrium:
    """Long-time simulation finds a sensible coexistence equilibrium."""

    def test_equilibrium_under_no_drug_returns_positive(self, default_sim: LV3PopKShift) -> None:
        """All three populations should be non-trivial at the equilibrium."""
        x_eq = default_sim.equilibrium_under_no_drug()
        assert x_eq.shape == (3,)
        assert np.all(x_eq > 0), f"Some population went extinct: x_eq={x_eq}"

    def test_equilibrium_satisfies_dynamics(self, default_sim: LV3PopKShift) -> None:
        """At the returned equilibrium, dynamics should be near zero
        (verified inside equilibrium_under_no_drug; this re-checks)."""
        x_eq = default_sim.equilibrium_under_no_drug()
        dx = default_sim.dynamics(0.0, x_eq, Lambda=0.0)
        rel_err = np.abs(dx) / np.maximum(np.abs(x_eq), 1.0)
        np.testing.assert_array_less(rel_err, 1e-2)

    def test_equilibrium_with_explicit_x0(self, default_sim: LV3PopKShift) -> None:
        """Different initial guesses should reach the same attractor (loosely).

        Per ``equilibrium_under_no_drug`` docstring, the system has
        multi-time-scale dynamics and T+ drifts slowly toward its true
        equilibrium. At the default t_max=5000, T+ may still differ by
        ~10-15% across initial conditions. The TP and T- components,
        which equilibrate faster, agree more tightly.
        """
        x_eq1 = default_sim.equilibrium_under_no_drug(x0=(1000.0, 5000.0, 1000.0))
        x_eq2 = default_sim.equilibrium_under_no_drug(x0=(500.0, 3000.0, 500.0))
        # T+ (index 0) tolerated 15% rel; TP and T- tighter at 5%.
        np.testing.assert_allclose(x_eq1[1:], x_eq2[1:], rtol=5e-2)
        np.testing.assert_allclose(x_eq1[0], x_eq2[0], rtol=15e-2)
