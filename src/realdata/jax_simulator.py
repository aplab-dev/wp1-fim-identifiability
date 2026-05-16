"""JAX-native LV3PopKShift simulator with AD-stable smooth floors.

Reimplements the LV3PopKShift dynamics + first-order PSA filter in pure
JAX via diffrax for NUTS-compatible per-patient inference.

Key implementation choice: **smooth softplus floor** for K_TP and K_T+.
Naïve `jnp.maximum(K, eps)` produces NaN gradients under reverse-mode
AD when state hits the floor (the `1 / K_i` divisions in the dynamics
have unbounded gradient as K_i → 0). The fix:

    K_smooth(K, eps) = 0.5 * (K + sqrt(K^2 + eps^2))

This approximates max(K, eps) for K >> eps and remains everywhere
differentiable for all K. Gradients stay bounded.

Solver: Heun (RK2) fixed-step at dt=0.5 days. The K-shift dynamics have
multi-time-scale structure that breaks adaptive solvers; fixed-step
explicit RK2 with smooth floors is stable forward AND under reverse-
mode AD. ~3000 steps over 1500 days, ~3 ms per JIT-compiled call.

Performance budget for NUTS at 500 samples × 2 chains × ~50 leapfrogs ×
~5 ms with gradient = ~250 s per patient. Acceptable for Phase 3 §3.3
clinical-grade fits (n=70 patients × 250 s ≈ 5 hours, parallelizable).

Used by ``per_patient_hmc.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Force x64 mode for numerical stability in PSA filter
jax.config.update("jax_enable_x64", True)

import diffrax  # noqa: E402 — must come after jax_enable_x64

from zhang2017 import ZHANG_CANONICAL_X0, zhang_canonical_lv_params  # noqa: E402

_canon = zhang_canonical_lv_params()


def _smooth_floor(x, eps=1e-3):
    """Smooth approximation of max(x, eps) that is everywhere differentiable.

    For x >> eps: ≈ x.  For x << -eps: ≈ 0 (but with positive value eps).
    For x ≈ 0: ≈ eps. Replaces jnp.maximum(x, eps) which has zero
    gradient on one side and NaN under reverse-mode AD when state
    crosses the boundary mid-trajectory.
    """
    return 0.5 * (x + jnp.sqrt(x * x + eps * eps))


def _build_alpha(theta):
    """Build a JAX 3x3 alpha matrix from the 6-parameter theta."""
    # Hold off-diagonal T+/TP block at canonical values.
    a01 = float(_canon.alpha[0, 1])
    a02 = float(_canon.alpha[0, 2])
    a10 = float(_canon.alpha[1, 0])
    a12 = float(_canon.alpha[1, 2])
    return jnp.array([
        [1.0, a01, a02],
        [a10, 1.0, a12],
        [theta[3], theta[4], 1.0],
    ])


def _rhs_3pop_psa(t, y, args):
    """JAX-native RHS of (T+, TP, T-, PSA) under piecewise-constant Λ schedule.

    args = (theta, t_grid, u_grid, psa_phi, psa_rho)

    State positivity is enforced via smooth-floor on the cell counts in the
    dynamics' multiplicative term. This prevents x_i from going negative
    during fixed-step integration even when the K-shift collapse drives a
    very-negative dx_i/dt for short times. Mathematically, this is a tiny
    soft "immigration term" that only activates near x=0; clinically
    irrelevant.
    """
    theta, t_grid, u_grid, psa_phi, psa_rho = args
    x_Tplus, x_TP, x_Tminus, psa = y[0], y[1], y[2], y[3]
    # Interpolate u(t): find largest index i s.t. t_grid[i] <= t.
    idx = jnp.clip(jnp.searchsorted(t_grid, t, side="right") - 1, 0, u_grid.shape[0] - 1)
    u = u_grid[idx]

    # Smooth state floors — keep cell counts positive for stable AD.
    x_Tplus_safe = _smooth_floor(x_Tplus, eps=1e-3)
    x_TP_safe = _smooth_floor(x_TP, eps=1e-3)
    x_Tminus_safe = _smooth_floor(x_Tminus, eps=1e-3)

    # K-shift formulae with smooth floors for AD stability.
    K_TP_raw = _canon.K_TP_max - theta[5] * u
    K_TP = _smooth_floor(K_TP_raw, eps=1.0)
    mu = _canon.mu_max - _canon.mu_drop * u
    K_Tplus_raw = mu * x_TP_safe
    K_Tplus = _smooth_floor(K_Tplus_raw, eps=1.0)
    K_Tminus_val = float(_canon.K_Tminus)

    # Competition matrix
    alpha = _build_alpha(theta)
    x_vec = jnp.array([x_Tplus_safe, x_TP_safe, x_Tminus_safe])
    comp = alpha @ x_vec

    # Dynamics — use safe state values to prevent multiplicative blow-up.
    r_Tplus = _smooth_floor(theta[0], eps=1e-6)
    r_TP = _smooth_floor(theta[1], eps=1e-6)
    r_Tminus = _smooth_floor(theta[2], eps=1e-6)
    dx_Tplus = r_Tplus * x_Tplus_safe * (K_Tplus - comp[0]) / K_Tplus
    dx_TP = r_TP * x_TP_safe * (K_TP - comp[1]) / K_TP
    dx_Tminus = r_Tminus * x_Tminus_safe * (K_Tminus_val - comp[2]) / K_Tminus_val

    # PSA filter
    total = x_Tplus_safe + x_TP_safe + x_Tminus_safe
    dpsa = psa_rho * total - psa_phi * psa

    return jnp.array([dx_Tplus, dx_TP, dx_Tminus, dpsa])


def _make_jax_predictor(t_obs, u_schedule, psa_phi=0.5, psa_rho=1.0):
    """Build a JIT-compiled function theta -> psa_at_t_obs.

    Returns a callable jax_predict_psa(theta) -> psa array of len(t_obs).
    The closure captures t_obs and u_schedule as static (must be re-built
    if those change).
    """
    t_grid = jnp.asarray(t_obs)
    u_grid = jnp.asarray(u_schedule)
    t_eval = jnp.asarray(t_obs)
    psa_phi_arr = float(psa_phi)
    psa_rho_arr = float(psa_rho)
    # Cache concrete endpoint times before tracing (used as static t0/t1).
    t0_static = float(t_obs[0])
    t1_static = float(t_obs[-1])

    # Initial state — fixed across patients per Zhang 2017 reassessment.
    psa0_init = psa_rho * float(sum(ZHANG_CANONICAL_X0)) / psa_phi
    y0 = jnp.array([
        float(ZHANG_CANONICAL_X0[0]),
        float(ZHANG_CANONICAL_X0[1]),
        float(ZHANG_CANONICAL_X0[2]),
        psa0_init,
    ])

    term = diffrax.ODETerm(_rhs_3pop_psa)
    # Adaptive Tsit5 with loose tolerances + checkpointed adjoint for AD.
    # Smooth-floor formulation makes the dynamics non-stiff for moderate
    # tolerances (rtol=1e-3, atol=1e-1). Adaptive stepping is FAR faster
    # under reverse-mode AD than fixed-step (which would store 3000 states
    # per backward pass). Typical patient: ~200 adaptive steps.
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=t_eval)
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-1)

    # Recursive checkpointed adjoint: store every Nth state, recompute the
    # rest during backward pass. Trades a small forward-pass cost for
    # massively reduced memory + time during reverse-mode AD.
    adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=32)

    def jax_predict_psa(theta):
        args = (theta, t_grid, u_grid, psa_phi_arr, psa_rho_arr)
        sol = diffrax.diffeqsolve(
            term, solver,
            t0=t0_static, t1=t1_static,
            dt0=1.0,
            y0=y0, args=args,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=50_000,
            adjoint=adjoint,
            throw=False,  # don't raise on step-limit; instead return partial solution
        )
        return sol.ys[:, 3]  # PSA column

    return jax.jit(jax_predict_psa)


def _make_jax_predictor_native(t_obs, u_schedule, psa_phi=0.5, psa_rho=1.0,
                                dt: float = 0.5):
    """JAX-native fixed-step Heun (RK2) integrator — no diffrax.

    This replaces the diffrax-based ``_make_jax_predictor`` for use under
    NUTS. The diffrax version triggered an unresolved warmup-hang on this
    Python/JAX/diffrax/numpyro stack when NUTS dual-averaging proposed
    extreme θ values that pushed the adaptive solver into its `max_steps`
    ceiling, causing the checkpointed adjoint to re-trace pathologically.

    This implementation:
    - Uses ``jax.lax.scan`` with a fixed-step Heun integrator (RK2).
    - Steps from ``t0`` to ``t1`` in increments of ``dt`` (default 0.5d).
    - At each step, interpolates the piecewise-constant drug schedule via
      ``searchsorted``.
    - Linearly interpolates the saved trajectory at observation times
      ``t_obs``.
    - Smooth-floor everywhere (no jnp.maximum) so reverse-mode AD is stable.
    - JIT-compiles end-to-end. No ``max_steps`` ceiling. No adjoint
      recomputation. Just a single forward+backward scan, suitable for NUTS.

    Performance (M-class CPU):
      - Forward JIT-compiled: ~3-4 ms / call for a 1500-day patient at dt=0.5.
      - Reverse-mode gradient: ~25-30 ms / call.
      - Comparable to diffrax but with bounded, predictable cost.

    Args mirror ``_make_jax_predictor``.

    Returns:
        Callable ``jax_predict_psa(theta) -> jnp.ndarray of shape (len(t_obs),)``.
    """
    t_grid_static = jnp.asarray(t_obs)
    u_grid_static = jnp.asarray(u_schedule)
    psa_phi_arr = float(psa_phi)
    psa_rho_arr = float(psa_rho)
    t0_static = float(t_obs[0])
    t1_static = float(t_obs[-1])

    # Initial state — same as diffrax version.
    psa0_init = psa_rho * float(sum(ZHANG_CANONICAL_X0)) / psa_phi
    y0 = jnp.array([
        float(ZHANG_CANONICAL_X0[0]),
        float(ZHANG_CANONICAL_X0[1]),
        float(ZHANG_CANONICAL_X0[2]),
        psa0_init,
    ])

    # Build the fixed-step time grid.
    n_steps = int(jnp.ceil(jnp.asarray((t1_static - t0_static) / dt)))
    n_steps = int(n_steps) + 1  # endpoint inclusive
    t_steps_static = jnp.asarray(t0_static + jnp.arange(n_steps) * dt)

    # Index lookup: for each obs time, find the largest step index t_steps[i] <= t_obs.
    obs_step_idx_static = jnp.clip(
        jnp.searchsorted(t_steps_static, t_grid_static, side="right") - 1,
        0, n_steps - 2,
    )
    # Fractional offset for linear interpolation between step i and step i+1.
    obs_step_frac_static = (t_grid_static - t_steps_static[obs_step_idx_static]) / dt

    def jax_predict_psa(theta):
        args = (theta, t_grid_static, u_grid_static, psa_phi_arr, psa_rho_arr)

        def rhs(t, y):
            return _rhs_3pop_psa(t, y, args)

        def heun_step(y, t):
            # y' at t with current y
            k1 = rhs(t, y)
            # Predictor step
            y_pred = y + dt * k1
            # Corrector at t + dt with predicted y
            k2 = rhs(t + dt, y_pred)
            # Heun update: average of k1 and k2
            return y + 0.5 * dt * (k1 + k2), y

        # Run scan to integrate forward, saving the state at each step.
        _, ys = jax.lax.scan(heun_step, y0, t_steps_static)
        # ys has shape (n_steps, 4) with the PSA in column 3.
        psa_trace = ys[:, 3]

        # Linearly interpolate PSA at observation times.
        psa_at_obs = (
            psa_trace[obs_step_idx_static] * (1.0 - obs_step_frac_static)
            + psa_trace[obs_step_idx_static + 1] * obs_step_frac_static
        )
        return psa_at_obs

    return jax.jit(jax_predict_psa)
