"""Cunningham 2020 optimal-control reproduction module — Phase 2 Stage 2.5.

Implements the L₂-to-target-equilibrium optimal-control problem from
Cunningham et al. 2020 [PLOS One 15(12):e0243386] using `casadi` +
`ipopt` for direct multiple-shooting collocation.

The problem (Cunningham 2020 §"Optimal control formulation"):

    minimize  ∫_0^T ||x(t) - x*||² dt
    subject to:
      dx_i/dt = r_i x_i (K_i(Λ; x_TP) - sum_j alpha_ij x_j) / K_i(Λ; x_TP),
                                                    i ∈ {T+, TP, T-}
      Λ(t) ∈ [0, 1]
      x(0) = x_0
      x(t) >= 0 for all t (positivity)

Cunningham reports T = 10,000 days. We use a shorter horizon (default
3000 days, ~100 months) for tractability while preserving the smooth-Λ
qualitative result.

Two target equilibria are typically tested:
- x*_a (small): a "stable adaptive" equilibrium with low resistance, low total burden.
- x*_b (large): a "controlled-but-large" equilibrium where total burden is high but stable.

Both use the no-drug equilibrium of the 3-pop K-shift model as the baseline,
modified by a chosen target Λ.

Exports:
- ``solve_cunningham_ocp(params, x0, target, t_max, n_intervals)``.
- ``CunninghamSolution`` — container for trajectory + control + objective value.

References:
- ``docs/literature/cunningham-2020-optimal-control.md`` (deep-read reassessment).
- ``docs/methodology/phase2_plan.md`` §3 Stage 2.5.
"""

from .ocp import (  # noqa: F401
    CunninghamSolution,
    cunningham_target_equilibrium,
    solve_cunningham_ocp,
)
