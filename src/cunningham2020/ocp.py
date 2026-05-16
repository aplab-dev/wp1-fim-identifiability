"""Cunningham 2020 optimal-control problem (OCP) solver — multiple-shooting via casadi + ipopt.

Direct multiple-shooting formulation:
- Discretize time into N control intervals of length dt.
- Control Λ_k is constant on each interval, k = 0, ..., N-1.
- State propagated through each interval via casadi RK4 integrator (8 sub-steps).
- Decision variables: x_0, x_1, ..., x_N (state at interval boundaries) + Λ_0, ..., Λ_{N-1}.
- Constraints: ODE shooting equality + positivity + control bounds.
- Objective: trapezoidal rule on ||x_k - x*||² + 0.0 control regularization.

Why multiple-shooting and not direct collocation? Multiple-shooting is simpler to implement,
robust to long horizons, and works well for smooth optimal-control problems where Λ is
expected to be slowly varying. casadi's auto-differentiation of the integrator handles
gradient propagation automatically.

The 3-pop K-shift dynamics (Eq. 6 in WP1):

    dx_i/dt = r_i x_i (K_i(Λ; x_TP) - sum_j alpha_ij x_j) / K_i(Λ; x_TP)

Some parameters and the alpha matrix come from the existing LV3PopParams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import casadi as ca
import numpy as np

from simulators.lv_3pop_kshift import LV3PopKShift, LV3PopParams


@dataclass
class CunninghamSolution:
    """Result of an OCP solve.

    Attributes:
        t: time grid (N+1 points, 0 to T_max).
        x: (3, N+1) state trajectory: rows = (T+, TP, T-).
        Lambda: length-N control trajectory (piecewise-constant on intervals).
        objective: optimized objective value.
        target: x* used for the L₂ objective.
        params: LV3PopParams used.
        ipopt_status: solver status string.
    """

    t: np.ndarray
    x: np.ndarray
    Lambda: np.ndarray
    objective: float
    target: np.ndarray
    params: LV3PopParams = field(repr=False)
    ipopt_status: str = "Solve_Succeeded"


def cunningham_target_equilibrium(
    params: LV3PopParams,
    target_Lambda: float = 0.4,
    x0_guess: tuple[float, float, float] | None = None,
    t_max_eq: float = 5_000.0,
) -> np.ndarray:
    """Compute the target equilibrium x* by simulating long-time forward at constant target_Lambda.

    Cunningham 2020 picks target equilibria of two types:
    - x*_a (small target): equilibrium under a moderate Lambda (~0.4), where T+/TP coexist
      and T- is suppressed. This is the "stable adaptive" attractor.
    - x*_b (large target): equilibrium near the no-drug case (Lambda → 0).

    For default settings, target_Lambda=0.4 typically gives x*_a (small attractor).
    """
    sim = LV3PopKShift(params)
    if x0_guess is None:
        x0_guess = (params.K_Tminus / 4, params.K_TP_max / 2, params.K_Tminus / 2)
    result = sim.simulate(
        x0=x0_guess, t_span=(0.0, t_max_eq), control=target_Lambda,
        t_eval=np.array([t_max_eq]),
    )
    return np.array([result.x_Tplus[-1], result.x_TP[-1], result.x_Tminus[-1]])


def _build_dynamics_function(params: LV3PopParams) -> ca.Function:
    """Build a casadi Function for the 3-pop K-shift RHS, suitable for integration."""
    x = ca.MX.sym("x", 3)  # (T+, TP, T-)
    u = ca.MX.sym("u")  # Lambda
    x_Tplus, x_TP, x_Tminus = x[0], x[1], x[2]

    # K-shift functions
    K_TP = ca.fmax(params.K_TP_max - params.K_TP_drop * u, 1e-6)
    mu = params.mu_max - params.mu_drop * u
    K_Tplus = ca.fmax(mu * x_TP, 1e-6)
    K_Tminus = params.K_Tminus

    # competition sum_j alpha_ij x_j for each i; alpha is 3x3 numeric.
    alpha_dm = ca.DM(params.alpha)
    comp = alpha_dm @ x  # 3x1

    dx_Tplus = params.r_Tplus * x_Tplus * (K_Tplus - comp[0]) / K_Tplus
    dx_TP = params.r_TP * x_TP * (K_TP - comp[1]) / K_TP
    dx_Tminus = params.r_Tminus * x_Tminus * (K_Tminus - comp[2]) / K_Tminus

    return ca.Function("rhs", [x, u], [ca.vertcat(dx_Tplus, dx_TP, dx_Tminus)])


def _make_rk4_integrator(rhs: ca.Function, dt: float, n_substeps: int = 8) -> ca.Function:
    """Build an RK4 integrator for one control interval of length dt."""
    h = dt / n_substeps
    x0 = ca.MX.sym("x0", 3)
    u = ca.MX.sym("u")
    x = x0
    for _ in range(n_substeps):
        k1 = rhs(x, u)
        k2 = rhs(x + h / 2 * k1, u)
        k3 = rhs(x + h / 2 * k2, u)
        k4 = rhs(x + h * k3, u)
        x = x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return ca.Function("integrator", [x0, u], [x])


def solve_cunningham_ocp(
    params: LV3PopParams,
    x0: tuple[float, float, float] | np.ndarray,
    target: np.ndarray,
    t_max: float = 3000.0,
    n_intervals: int = 60,
    Lambda_init: float | np.ndarray = 0.0,
    control_smoothness_weight: float = 0.0,
    verbose: bool = False,
    ipopt_max_iter: int = 200,
) -> CunninghamSolution:
    """Solve the Cunningham 2020 L₂-to-equilibrium OCP via multiple-shooting.

    Args:
        params: 3-pop K-shift parameter set.
        x0: initial state (T+, TP, T-).
        target: target equilibrium x*.
        t_max: time horizon (days). Default 3000 d (~100 mo). Cunningham used 10000 d.
        n_intervals: number of control intervals. Higher = finer Λ resolution.
        Lambda_init: initial guess for Λ. Scalar or length-N array.
        control_smoothness_weight: optional penalty on (Λ_k - Λ_{k-1})² to
            prefer smoother controls (default 0 = no penalty).
        verbose: print ipopt progress.
        ipopt_max_iter: max ipopt iterations.

    Returns:
        CunninghamSolution.

    Raises:
        RuntimeError: if ipopt fails.
    """
    x0_np = np.asarray(x0, dtype=float)
    target_np = np.asarray(target, dtype=float)
    if x0_np.shape != (3,) or target_np.shape != (3,):
        raise ValueError("x0 and target must be length-3")

    dt = t_max / n_intervals
    rhs = _build_dynamics_function(params)
    integrator = _make_rk4_integrator(rhs, dt, n_substeps=8)

    opti = ca.Opti()

    # Decision variables.
    X = opti.variable(3, n_intervals + 1)  # state at each boundary
    U = opti.variable(n_intervals)  # piecewise-constant control

    # Initial-condition constraint
    opti.subject_to(X[:, 0] == x0_np)

    # Multiple-shooting equality constraints + positivity bounds + control bounds.
    objective = 0.0
    target_dm = ca.DM(target_np)
    for k in range(n_intervals):
        x_next = integrator(X[:, k], U[k])
        opti.subject_to(X[:, k + 1] == x_next)
        opti.subject_to(X[:, k + 1] >= 0)
        opti.subject_to(opti.bounded(0.0, U[k], 1.0))
        # Trapezoidal-rule running cost.
        cost_k = ca.sumsqr(X[:, k] - target_dm)
        cost_kp1 = ca.sumsqr(X[:, k + 1] - target_dm)
        objective = objective + dt * 0.5 * (cost_k + cost_kp1)
        if control_smoothness_weight > 0 and k > 0:
            objective = objective + control_smoothness_weight * (U[k] - U[k - 1]) ** 2

    opti.minimize(objective)

    # Initial guesses.
    if isinstance(Lambda_init, (int, float)):
        Lambda_init_arr = np.full(n_intervals, float(Lambda_init))
    else:
        Lambda_init_arr = np.asarray(Lambda_init, dtype=float)
        if Lambda_init_arr.shape != (n_intervals,):
            raise ValueError(f"Lambda_init array must be length {n_intervals}")
    opti.set_initial(U, Lambda_init_arr)

    # State initial guess: forward-simulate at Lambda_init_arr[0].
    X_init = np.zeros((3, n_intervals + 1))
    X_init[:, 0] = x0_np
    cur = x0_np.copy()
    for k in range(n_intervals):
        try:
            x_next_dm = integrator(cur, Lambda_init_arr[k])
            x_next_arr = np.array(x_next_dm).flatten()
            cur = np.maximum(x_next_arr, 1e-3)
        except Exception:  # noqa: BLE001
            cur = np.array(target_np)  # fallback
        X_init[:, k + 1] = cur
    opti.set_initial(X, X_init)

    # ipopt options.
    p_opts = {"print_time": int(verbose)}
    s_opts = {
        "print_level": 5 if verbose else 0,
        "max_iter": ipopt_max_iter,
        "tol": 1e-6,
        "acceptable_tol": 1e-4,
        "linear_solver": "mumps",
    }
    opti.solver("ipopt", p_opts, s_opts)

    try:
        sol = opti.solve()
        status = "Solve_Succeeded"
    except RuntimeError as e:
        # Try to recover a feasible-near-optimal solution.
        try:
            X_val = opti.debug.value(X)
            U_val = opti.debug.value(U)
            obj_val = opti.debug.value(objective)
            status = f"Solve_Failed_Recovered: {str(e)[:200]}"
        except Exception:
            raise RuntimeError(f"ipopt failed and could not recover: {e}") from e
    else:
        X_val = sol.value(X)
        U_val = sol.value(U)
        obj_val = float(sol.value(objective))

    t_grid = np.linspace(0, t_max, n_intervals + 1)
    return CunninghamSolution(
        t=t_grid,
        x=np.array(X_val),
        Lambda=np.atleast_1d(np.asarray(U_val)),
        objective=float(obj_val),
        target=target_np,
        params=params,
        ipopt_status=status,
    )
