"""Heuristic and learned policies for adaptive cancer therapy.

Phase 2 implementation (started 2026-05-02 session 16, Stage 2.3).

A *policy* is a function that maps the current observation (cell counts
and/or PSA) to a treatment decision (Λ ∈ [0, 1] for the K-shift simulator,
or u ∈ [0, 1] for the multdeath simulator).

The shared interface is :class:`Policy` (in ``base``) — a stateful object
with a ``__call__(observation)`` method. Policies are stateful because
adaptive protocols (AT50) need to remember whether they're currently in
a drug-on or drug-off phase.

For non-stateful policies (MTD, no-treatment), we still wrap them in the
Policy interface for uniformity. This makes the cohort runner simpler.

Currently exposed:
- ``base.Policy`` — abstract base class.
- ``mtd.MTDPolicy`` — always-on.
- ``no_treatment.NoTreatmentPolicy`` — always-off.
- ``at50.AT50Policy`` — Zhang 2017 protocol: drug on until 50% PSA decline;
  off until PSA returns to baseline.
"""

from .at50 import AT50Policy  # noqa: F401
from .base import Observation, Policy, PolicyState  # noqa: F401
from .cohort_runner import CohortResult, CohortRunner  # noqa: F401
from .mtd import MTDPolicy  # noqa: F401
from .no_treatment import NoTreatmentPolicy  # noqa: F401
