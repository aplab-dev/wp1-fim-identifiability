"""Real-data fitting layer — Phase 3 §3.3 / Month 7 deliverable.

Currently provides:
- ``bruchovsky.py`` — schema definitions + synthetic-cohort generator that
  mimics the Bruchovsky 2008 IADT cohort structure (n=70 patients with
  per-patient PSA trajectories on intermittent ADT). Real data ingestion
  is wired through the same schema; supplying actual Bruchovsky 2008
  PSA series will work as a drop-in replacement.
- ``per_patient_mcmc.py`` — Bayesian inference layer per patient. Adaptive
  Metropolis-Hastings on the 6-parameter 3-pop K-shift model with a
  Gaussian likelihood + weak prior. Returns posterior samples + diagnostic
  R-hat across multiple chains.

These together implement the core M7 deliverable from
``docs/methodology/phase3_skeleton.md`` §3.3.
"""

from .bruchovsky import (  # noqa: F401
    BruchovskyPatient,
    BruchovskyCohort,
    generate_synthetic_cohort,
    load_cohort_csv,
    load_dataTanaka,
    load_shaw_et_al,
)
from .per_patient_mcmc import (  # noqa: F401
    MCMCResult,
    fit_patient_mcmc,
    rhat_split,
)
from .hierarchical import (  # noqa: F401
    HierarchicalFit,
    PatientSummary,
    compare_pooled_vs_unpooled,
    hierarchical_fit,
    per_patient_summaries,
)
