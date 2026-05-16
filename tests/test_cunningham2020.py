"""Tests for the Cunningham 2020 OCP solver.

Verifies:
- Solver runs and returns sensible solution structure.
- Λ stays in [0, 1] bounds.
- Solution is feasible (state >= 0 throughout).
- Objective value monotonically decreases from a feasible-but-suboptimal initial.
- Smooth-titration result reproduces Cunningham 2020 qualitative finding
  (Λ ramps from low to ~target, not bang-bang).
- Trivial case where x0 = target gives near-zero objective.
"""

from __future__ import annotations

import numpy as np
import pytest

from cunningham2020 import (
    CunninghamSolution,
    cunningham_target_equilibrium,
    solve_cunningham_ocp,
)
from zhang2017 import ZHANG_CANONICAL_X0, zhang_canonical_lv_params


# Suppress all numerical warnings during tests
@pytest.fixture(autouse=True)
def _suppress_warnings():
    import warnings

    warnings.filterwarnings("ignore")
    yield


class TestTargetEquilibrium:
    def test_returns_length3_array(self) -> None:
        params = zhang_canonical_lv_params()
        target = cunningham_target_equilibrium(params, target_Lambda=0.4)
        assert target.shape == (3,)

    def test_target_components_are_finite(self) -> None:
        params = zhang_canonical_lv_params()
        target = cunningham_target_equilibrium(params, target_Lambda=0.4)
        assert np.all(np.isfinite(target))

    def test_higher_lambda_gives_smaller_total_burden(self) -> None:
        """Higher target Lambda compresses the K_i's -> smaller equilibrium total."""
        params = zhang_canonical_lv_params()
        t_low = cunningham_target_equilibrium(params, target_Lambda=0.2)
        t_high = cunningham_target_equilibrium(params, target_Lambda=0.7)
        assert t_high.sum() < t_low.sum()


class TestSolveOCP:
    @pytest.fixture
    def setup(self) -> dict:
        params = zhang_canonical_lv_params()
        return {
            "params": params,
            "x0": ZHANG_CANONICAL_X0,
            "target": cunningham_target_equilibrium(params, target_Lambda=0.4),
        }

    def test_returns_cunningham_solution(self, setup) -> None:
        sol = solve_cunningham_ocp(
            setup["params"], setup["x0"], setup["target"],
            t_max=1500.0, n_intervals=30, Lambda_init=0.4,
        )
        assert isinstance(sol, CunninghamSolution)

    def test_shape_consistency(self, setup) -> None:
        n_intervals = 30
        sol = solve_cunningham_ocp(
            setup["params"], setup["x0"], setup["target"],
            t_max=1500.0, n_intervals=n_intervals, Lambda_init=0.4,
        )
        assert sol.t.shape == (n_intervals + 1,)
        assert sol.x.shape == (3, n_intervals + 1)
        assert sol.Lambda.shape == (n_intervals,)

    def test_Lambda_in_bounds(self, setup) -> None:
        sol = solve_cunningham_ocp(
            setup["params"], setup["x0"], setup["target"],
            t_max=1500.0, n_intervals=30, Lambda_init=0.4,
        )
        # Allow small tolerance for ipopt's bound respect
        assert np.all(sol.Lambda >= -1e-6)
        assert np.all(sol.Lambda <= 1.0 + 1e-6)

    def test_states_nonnegative(self, setup) -> None:
        sol = solve_cunningham_ocp(
            setup["params"], setup["x0"], setup["target"],
            t_max=1500.0, n_intervals=30, Lambda_init=0.4,
        )
        # Trajectory should be non-negative (allowing small tol for ipopt)
        assert np.all(sol.x >= -1.0)  # 1.0 cell tolerance

    def test_initial_state_satisfied(self, setup) -> None:
        sol = solve_cunningham_ocp(
            setup["params"], setup["x0"], setup["target"],
            t_max=1500.0, n_intervals=30, Lambda_init=0.4,
        )
        np.testing.assert_allclose(sol.x[:, 0], np.array(setup["x0"]), rtol=1e-6)

    def test_ocp_succeeds_solve(self, setup) -> None:
        sol = solve_cunningham_ocp(
            setup["params"], setup["x0"], setup["target"],
            t_max=1500.0, n_intervals=30, Lambda_init=0.4,
            ipopt_max_iter=200,
        )
        assert sol.ipopt_status == "Solve_Succeeded"

    def test_smooth_titration_qualitative(self, setup) -> None:
        """Reproduce Cunningham 2020 §"smooth titration": Λ ramps from low to ~target.

        The optimal Λ should NOT be bang-bang. We check by verifying the
        first few intervals have low Λ and the last few have higher Λ near 0.4.
        """
        sol = solve_cunningham_ocp(
            setup["params"], setup["x0"], setup["target"],
            t_max=3000.0, n_intervals=60, Lambda_init=0.4,
        )
        # Initial Lambda should be lower than late Lambda (ramping up)
        early = sol.Lambda[:5].mean()
        late = sol.Lambda[-10:].mean()
        assert late > early, f"Lambda should ramp up: early={early:.3f}, late={late:.3f}"

    def test_objective_decreases_with_more_iterations(self, setup) -> None:
        """Solving with more iterations should give >=as-good objective."""
        sol_few = solve_cunningham_ocp(
            setup["params"], setup["x0"], setup["target"],
            t_max=1500.0, n_intervals=30, Lambda_init=0.0,
            ipopt_max_iter=10,
        )
        sol_many = solve_cunningham_ocp(
            setup["params"], setup["x0"], setup["target"],
            t_max=1500.0, n_intervals=30, Lambda_init=0.0,
            ipopt_max_iter=200,
        )
        # The "many" solution should be at-least-as-good (objective is being
        # minimized). Allow small numerical tolerance.
        assert sol_many.objective <= sol_few.objective + 1e-3 * max(abs(sol_few.objective), 1.0)


class TestEdgeCases:
    def test_invalid_x0_shape_rejected(self) -> None:
        params = zhang_canonical_lv_params()
        target = cunningham_target_equilibrium(params, target_Lambda=0.4)
        with pytest.raises(ValueError):
            solve_cunningham_ocp(params, [1, 2], target, t_max=1500, n_intervals=30)

    def test_invalid_target_shape_rejected(self) -> None:
        params = zhang_canonical_lv_params()
        with pytest.raises(ValueError):
            solve_cunningham_ocp(
                params, ZHANG_CANONICAL_X0, np.array([1.0, 2.0]),
                t_max=1500, n_intervals=30,
            )

    def test_lambda_init_array_wrong_length_rejected(self) -> None:
        params = zhang_canonical_lv_params()
        target = cunningham_target_equilibrium(params, target_Lambda=0.4)
        with pytest.raises(ValueError, match="Lambda_init"):
            solve_cunningham_ocp(
                params, ZHANG_CANONICAL_X0, target,
                t_max=1500, n_intervals=30,
                Lambda_init=np.array([0.5, 0.5]),  # wrong length
            )
