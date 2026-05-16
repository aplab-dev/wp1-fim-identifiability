"""MTD (Maximum Tolerated Dose) — always-on continuous treatment.

The standard-of-care policy in most cytotoxic chemotherapy. Drug at full dose
from t=0 onward, no holidays. The baseline against which adaptive protocols
are typically compared.
"""

from __future__ import annotations

from .base import Observation, Policy


class MTDPolicy(Policy):
    """Always-on at full dose."""

    name: str = "MTD"

    def decide(self, obs: Observation) -> float:
        return 1.0
