"""Zhang 2017 reproduction module — Phase 2 Stage 2.4 primary target.

Wires the 3-pop K-shift simulator + PSA filter + heuristic policies
(Stages 2.1-2.3) into the Zhang 2017 mCRPC pilot-trial setup:

- **Per-patient initial conditions** at 25% of the untreated equilibrium,
  with each-patient log-normal IC perturbations to generate cohort variation
  (our choice; documented in `runner.run_zhang_patient`'s docstring — Zhang
  2017 itself does not perturb ICs).
- **PSA baseline** measured at the IC (i.e., baseline PSA = quasi-steady-state
  PSA for the patient's IC, NOT for the equilibrium total).
- **Progression** = first time PSA crosses ``progression_psa_threshold *
  baseline_psa`` (default 1.2× baseline, mirroring the practical Zhang 2017
  decision-rule for clinical PSA progression).
- **Decision cadence** = 28 days (4-week labs, per Zhang 2017 protocol).

Exports:
- ``ZhangPatientParams`` — per-patient parameter object.
- ``zhang_2017_sampler(rng)`` — sampler returning canonical ZhangPatientParams.
- ``run_zhang_patient(params, policy, rng=None)`` — single-patient runner
  compatible with ``policies.cohort_runner.RunOnePatient`` Protocol.

References:
- ``docs/literature/zhang-2017-crpc-adaptive.md`` (deep-read reassessment).
- ``docs/methodology/phase2_plan.md`` §3 Stage 2.4.
"""

from .runner import (  # noqa: F401
    ZHANG_CANONICAL_X0,
    ZhangPatientParams,
    run_zhang_patient,
    zhang_2017_sampler,
    zhang_canonical_lv_params,
)
