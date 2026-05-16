"""No-treatment control policy.

Useful as a sanity-check baseline: a tumor under no drug should grow
toward its untreated equilibrium (Derivation 1 §1.2), establishing the
counterfactual for any treatment claim.
"""

from __future__ import annotations

from .base import Observation, Policy


class NoTreatmentPolicy(Policy):
    """Always-off — no drug ever."""

    name: str = "No treatment"

    def decide(self, obs: Observation) -> float:
        return 0.0
