"""Single-patient runner for Zhang 2017 reproduction.

Wires together:
- ``simulators.lv_3pop_kshift.LV3PopKShift`` — 3-pop K-shift dynamics.
- ``simulators.psa_dynamics`` — first-order PSA filter.
- A ``policies.base.Policy`` — MTD / no-treatment / AT50 / etc.

Approach: simulate forward in `decision_interval`-day chunks (default 28 d
for clinical 4-week labs). At the start of each chunk, query the policy with
the current PSA observation. Hold the resulting drug level constant for the
chunk's duration. Concatenate chunk trajectories. Detect progression via
PSA threshold crossing.

The runner is compatible with ``policies.cohort_runner.RunOnePatient``:
``run_zhang_patient(params, policy, rng=None) -> dict``.

Cohort variation:
The Zhang 2017 paper itself uses a single canonical parameter set + uniform
ICs (25% of untreated equilibrium for all patients). To generate cohort
variation for a meaningful comparison plot, we add log-normal noise to the
IC vector with `ic_perturbation_std` (default 0.10 = 10% per-component).
This is **our modeling choice, not Zhang 2017's**, and is documented as
such in any figures or summary statistics that come out of the cohort
runner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from policies.base import Observation, Policy
from simulators.lv_3pop_kshift import LV3PopKShift, LV3PopParams
from simulators.psa_dynamics import PSAParams, psa_steady_state

log = logging.getLogger(__name__)


def zhang_canonical_lv_params() -> LV3PopParams:
    """LV3PopParams tuned for Zhang 2017 reproduction.

    Default LV3PopParams uses an illustrative-midpoint alpha matrix that
    happens to produce a T--dominated equilibrium — wrong for Zhang. This
    factory overrides alpha so T- is heavily suppressed by T+/TP at no-drug
    equilibrium (alpha[T-,T+]=3, alpha[T-,TP]=4 on a diagonal of 1). The
    rest of the parameters (growth rates, K-shift coefficients) match the
    simulator default, which is itself Cunningham 2020-derived.

    With this alpha:
    - Under no drug: T- is competitively excluded; T+/TP coexist around
      ~7000 each, T- → 0.
    - Under MTD: T+/TP collapse (K-shift), competitive release on T-,
      T- grows toward K_T-=10000.

    This is the simplest alpha matrix that produces Zhang 2017-like
    dynamics (PSA dropping deeply under MTD, AT50 cycling possible).
    Zhang 2017's actual supplementary table 1 lists 22 distinct
    rank-orderings; we use one that satisfies the same qualitative
    structure (T- suppressed in no-drug regime, released under treatment).
    """
    alpha = np.array([
        [1.0, 0.5, 0.7],   # row T+: effects on T+
        [0.4, 1.0, 0.5],   # row TP: effects on TP
        [3.0, 4.0, 1.0],   # row T-: effects on T- (heavily suppressed)
    ])
    return LV3PopParams(alpha=alpha)


# Canonical Zhang-2017-style initial condition: T+/TP dominate, T- a small
# resistance reservoir (~0.7% of total). Total ~7050 cells produces
# baseline PSA = 14100 with default PSAParams (phi=0.5, rho=1.0).
ZHANG_CANONICAL_X0: tuple[float, float, float] = (3500.0, 3500.0, 50.0)


@dataclass(frozen=True)
class ZhangPatientParams:
    """Per-patient parameter container for Zhang 2017 reproduction.

    Attributes:
        lv_params: LV3PopParams. Default = ``zhang_canonical_lv_params()``
            (Zhang-tuned alpha, NOT the simulator default).
        psa_params: PSAParams (Zhang 2017 default: phi=0.5/day, rho=1.0).
        x0: Initial condition (T+, TP, T-). Default = ZHANG_CANONICAL_X0
            = (3500, 3500, 50). Matches Zhang 2017's "ESS at 25% of
            untreated equilibrium" qualitatively (T+/TP dominate, T- ~1%).
        ic_perturbation_std: Std-dev of log-normal noise applied to x0
            components per-patient (only when rng is supplied). 0.0 means
            uniform IC across patients. Zhang 2017 itself uses uniform ICs;
            we add ic_perturbation_std=0.10 by default to generate cohort
            variation for plotting.
        progression_psa_threshold: Multiplier of baseline_psa above which
            the patient is considered to have progressed. Zhang 2017 uses
            1.20× baseline as a practical PSA-progression definition.
        t_max: Maximum simulation horizon (days). Default 1500 d ≈ 50 months,
            roughly matching Zhang 2017's follow-up window.
        decision_interval: Days between policy queries. Zhang 2017 protocol:
            4-week labs (28 days).
    """

    lv_params: LV3PopParams = field(default_factory=zhang_canonical_lv_params)
    psa_params: PSAParams = field(default_factory=PSAParams)
    x0: tuple[float, float, float] = ZHANG_CANONICAL_X0
    ic_perturbation_std: float = 0.10
    progression_psa_threshold: float = 1.20
    t_max: float = 1500.0
    decision_interval: float = 28.0


def zhang_2017_sampler(rng: np.random.Generator) -> ZhangPatientParams:  # noqa: ARG001
    """Default cohort sampler — returns the canonical Zhang patient.

    All patients in the cohort receive the same canonical parameters and the
    same default IC settings. Per-patient variation is introduced inside
    ``run_zhang_patient`` via the ``ic_perturbation_std`` mechanism using the
    runner's RNG (which CohortRunner spawns deterministically per patient).

    The rng arg is unused here but kept for Protocol compliance.
    """
    return ZhangPatientParams()


def _detect_progression(t: np.ndarray, psa: np.ndarray, threshold: float) -> tuple[float, bool]:
    """Find first time PSA crosses `threshold`. Return (ttp, progressed).

    If never crosses, returns (t[-1], False) — TTP capped at simulation
    horizon (right-censored).
    """
    crossings = np.where(psa >= threshold)[0]
    if crossings.size == 0:
        return float(t[-1]), False
    return float(t[crossings[0]]), True


def run_zhang_patient(
    params: ZhangPatientParams,
    policy: Policy,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Simulate one Zhang 2017 patient under a policy.

    Returns a dict with the keys required by ``CohortRunner.RunOnePatient``:
    - ``ttp``: time to progression (days), or t_max if right-censored.
    - ``cumulative_dose``: integral of Λ(t) over the run (drug-days).
    - ``progressed``: bool — did PSA cross the progression threshold.
    - ``baseline_psa``: pre-treatment PSA value (for diagnostic plots).
    - ``trajectory``: dict with t, x_Tplus, x_TP, x_Tminus, Lambda, psa
      arrays — useful for debugging and per-patient figures. Set to None
      if you don't need it (see ``include_trajectory`` knob below).

    Approach:
    1. Compute untreated equilibrium x_eq via ``equilibrium_under_no_drug``.
    2. Set IC = 0.25 * x_eq (per Zhang 2017 reassessment Correction 4).
       If ic_perturbation_std > 0 and rng provided, multiply each component
       by exp(N(0, ic_perturbation_std)) for cohort variation.
    3. Compute baseline_psa = psa_steady_state(sum(IC), psa_params).
    4. Loop in decision_interval-day chunks. Query policy with current PSA.
       Hold Λ constant for the chunk. Integrate (T+, TP, T-, PSA) jointly.
    5. Stop early if progression detected; otherwise run to t_max.
    """
    sim = LV3PopKShift(params.lv_params)

    # --- Initial condition ---
    x0 = np.array(params.x0, dtype=float)
    if params.ic_perturbation_std > 0 and rng is not None:
        log_noise = rng.normal(0.0, params.ic_perturbation_std, size=3)
        x0 = x0 * np.exp(log_noise)
    if np.any(x0 <= 0):
        raise ValueError(f"Computed IC has non-positive component: {x0}")

    baseline_psa = psa_steady_state(float(np.sum(x0)), params.psa_params)
    progression_psa = params.progression_psa_threshold * baseline_psa

    # --- Storage ---
    t_chunks = [np.array([0.0])]
    x_chunks = [x0.reshape(3, 1)]
    psa_chunks = [np.array([baseline_psa])]
    u_chunks = [np.array([0.0])]  # placeholder; first decision overwrites

    current = x0.copy()
    current_psa = baseline_psa
    t_now = 0.0
    cum_dose = 0.0
    ttp = params.t_max
    progressed = False

    while t_now < params.t_max:
        # Policy decides on current PSA observation.
        obs = Observation(t=t_now, psa=current_psa, baseline_psa=baseline_psa)
        u = float(policy(obs))

        t_end = min(t_now + params.decision_interval, params.t_max)

        # Coupled integration of (T+, TP, T-, PSA) over the chunk.
        psa_p = params.psa_params

        def rhs(t: float, y: np.ndarray, u_const: float = u) -> np.ndarray:
            x = y[:3]
            psa = y[3]
            dx = sim.dynamics(t, x, u_const)
            if psa_p.weights is not None:
                weighted = float(psa_p.weights @ x)
            else:
                weighted = float(np.sum(x))
            dpsa = psa_p.rho * weighted - psa_p.phi * psa
            return np.concatenate([dx, [dpsa]])

        y0 = np.concatenate([current, [current_psa]])
        # LSODA for stiffness handling under MTD (T+ collapse). Some
        # patient ICs trigger LSODA's "repeated convergence failures";
        # fall back to BDF (which handles stiff systems robustly).
        sol = None
        last_msg = ""
        for method in ("LSODA", "BDF"):
            try:
                trial = solve_ivp(
                    rhs,
                    t_span=(t_now, t_end),
                    y0=y0,
                    t_eval=np.linspace(t_now, t_end, 30),
                    method=method,
                    rtol=1e-6,
                    atol=1e-3,
                )
                if trial.success:
                    sol = trial
                    break
                last_msg = trial.message
            except Exception as e:  # noqa: BLE001 — solver-specific failures
                last_msg = str(e)
        if sol is None or not sol.success:
            raise RuntimeError(
                f"Zhang patient solve_ivp failed (LSODA + BDF): {last_msg}"
            )

        # Append (skip first point to avoid duplication with previous chunk's last).
        t_chunks.append(sol.t[1:])
        x_chunks.append(sol.y[:3, 1:])
        psa_chunks.append(sol.y[3, 1:])
        u_chunks.append(np.full(sol.t.size - 1, u))

        # Progression check on this chunk's PSA trajectory.
        chunk_t = sol.t
        chunk_psa = sol.y[3]
        ttp_chunk, prog_chunk = _detect_progression(chunk_t, chunk_psa, progression_psa)
        if prog_chunk:
            # Drug exposure only up to TTP, not the full chunk.
            cum_dose += u * (ttp_chunk - t_now)
            ttp = ttp_chunk
            progressed = True
            # Update state up to TTP for trajectory return (still keep the
            # full chunk integration; ttp is the reported endpoint).
            current = sol.y[:3, -1].copy()
            current_psa = float(sol.y[3, -1])
            t_now = t_end
            break

        # Update state for next iteration (no progression in this chunk).
        current = sol.y[:3, -1].copy()
        current_psa = float(sol.y[3, -1])
        cum_dose += u * (t_end - t_now)
        t_now = t_end

    t_full = np.concatenate(t_chunks)
    x_full = np.concatenate(x_chunks, axis=1)
    psa_full = np.concatenate(psa_chunks)
    u_full = np.concatenate(u_chunks)

    return {
        "ttp": ttp,
        "cumulative_dose": cum_dose,
        "progressed": progressed,
        "baseline_psa": baseline_psa,
        "trajectory": {
            "t": t_full,
            "x_Tplus": x_full[0],
            "x_TP": x_full[1],
            "x_Tminus": x_full[2],
            "Lambda": u_full,
            "psa": psa_full,
        },
    }
