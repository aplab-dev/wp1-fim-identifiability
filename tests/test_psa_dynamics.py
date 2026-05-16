"""Tests for the PSA dynamics filter.

Verifies:
- Steady-state PSA = rho * total / phi.
- Filter response time matches phi (e.g., decay half-life ln2/phi).
- Standalone integration on a known cell-count trajectory matches an
  analytical solution.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulators.psa_dynamics import PSAParams, integrate_psa_from_cells, psa_steady_state


# ---------- parameter validation ----------

class TestPSAParams:
    """PSAParams rejects invalid inputs."""

    def test_zero_phi_rejected(self) -> None:
        with pytest.raises(ValueError, match="phi"):
            PSAParams(phi=0.0)

    def test_negative_phi_rejected(self) -> None:
        with pytest.raises(ValueError, match="phi"):
            PSAParams(phi=-0.1)

    def test_negative_rho_rejected(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            PSAParams(rho=-0.5)

    def test_negative_weights_rejected(self) -> None:
        with pytest.raises(ValueError, match="weights"):
            PSAParams(weights=np.array([1.0, -0.1, 0.5]))

    def test_default_zhang_2017(self) -> None:
        """Defaults match Zhang 2017: phi = 0.5/day, rho = 1."""
        p = PSAParams()
        assert p.phi == pytest.approx(0.5)
        assert p.rho == pytest.approx(1.0)


# ---------- steady state ----------

class TestSteadyState:
    """psa_steady_state(total, params) = rho * total / phi."""

    def test_zhang_2017_defaults(self) -> None:
        # Total = 5000 cells, phi = 0.5, rho = 1 => SS = 10000
        p = PSAParams()
        ss = psa_steady_state(5000.0, p)
        assert ss == pytest.approx(10_000.0)

    def test_brady_nicholls_defaults(self) -> None:
        """Brady-Nicholls 2020 phi = 0.0856 (half-life ~8 d)."""
        p = PSAParams(phi=0.0856)
        ss = psa_steady_state(5000.0, p)
        assert ss == pytest.approx(5000.0 / 0.0856)


# ---------- standalone integration ----------

class TestStandaloneIntegration:
    """integrate_psa_from_cells matches analytic solutions."""

    def test_constant_cells_converges_to_steady_state(self) -> None:
        """If cells are held constant, PSA settles at the steady state."""
        p = PSAParams(phi=0.5, rho=1.0)
        n_pop = 2
        n_t = 200
        # Long enough that PSA equilibrates: 10/phi = 20 days
        t = np.linspace(0.0, 100.0, n_t)
        cells = np.full((n_pop, n_t), 100.0)  # constant 200 total per cell
        psa = integrate_psa_from_cells(t, cells, p, psa0=0.0)
        ss = psa_steady_state(200.0, p)  # 200 / 0.5 = 400
        # By t=100, with phi=0.5, we should be way past steady state (~50 half-lives)
        assert psa[-1] == pytest.approx(ss, rel=1e-3)

    def test_zero_cells_decays_to_zero(self) -> None:
        """With no cells, PSA should decay exponentially."""
        p = PSAParams(phi=0.5)
        t = np.linspace(0.0, 20.0, 100)
        cells = np.zeros((1, len(t)))
        psa = integrate_psa_from_cells(t, cells, p, psa0=100.0)
        # PSA(t) = 100 * exp(-0.5 * t)
        expected = 100.0 * np.exp(-0.5 * t)
        np.testing.assert_allclose(psa, expected, rtol=1e-3)

    def test_psa_uses_weights_when_provided(self) -> None:
        """With weights = (1, 0.1, 0.5), only weighted sum drives PSA."""
        weights = np.array([1.0, 0.1, 0.5])
        p = PSAParams(phi=0.5, rho=1.0, weights=weights)
        t = np.linspace(0.0, 100.0, 100)
        cells = np.full((3, len(t)), 100.0)  # 100 of each population
        psa = integrate_psa_from_cells(t, cells, p, psa0=0.0)
        # Effective total (weighted) = 100*1 + 100*0.1 + 100*0.5 = 160
        ss = psa_steady_state(160.0, p)
        assert psa[-1] == pytest.approx(ss, rel=1e-3)

    def test_callable_cells_supported(self) -> None:
        """A callable returning cell-count vector at any t should work."""
        p = PSAParams(phi=0.5)
        t = np.linspace(0.0, 50.0, 100)

        def cells_fn(tau: float) -> np.ndarray:
            return np.array([100.0, 200.0])

        psa = integrate_psa_from_cells(t, cells_fn, p, psa0=0.0)
        # Total = 300, SS = 600
        assert psa[-1] == pytest.approx(600.0, rel=1e-3)

    def test_wrong_array_shape_rejected(self) -> None:
        """Array with wrong N_t dimension should be rejected."""
        p = PSAParams()
        t = np.linspace(0.0, 10.0, 50)
        bad_cells = np.zeros((2, 100))  # N_t mismatch
        with pytest.raises(ValueError, match="cells_by_t"):
            integrate_psa_from_cells(t, bad_cells, p)
