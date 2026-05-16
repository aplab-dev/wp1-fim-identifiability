"""Policy base class and shared types.

A Policy is a stateful object that maps observations to drug levels. Stateful
because adaptive protocols (e.g., AT50) need to track whether they're currently
in drug-on or drug-off phase.

The Observation is a structured record passed to the policy at each decision
point — typically at clinical-appointment intervals (every 4-12 weeks in
clinical practice; every dt days in simulation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Observation:
    """What a policy sees at one decision point.

    Attributes:
        t: Current time (days since treatment start).
        psa: Serum PSA at time t. None if not measured this step.
        cells: Optional cell-count vector (S, R) for 2-pop or (T+, TP, T-) for
            3-pop. None if the policy operates only on PSA. The simulator
            knows the cell counts; the policy may or may not.
        baseline_psa: Pre-treatment PSA (set once at trial enrollment).
            Used by AT50 to compute the 50%-decline threshold.
    """

    t: float
    psa: float | None = None
    cells: np.ndarray | None = None
    baseline_psa: float | None = None


@dataclass
class PolicyState:
    """Mutable state carried by a stateful policy.

    Concrete policies subclass or extend this. Shared fields:

    - last_decision: the drug level returned at the last call.
    - n_decisions: how many times the policy has been called.
    - history: optional list of (Observation, decision) tuples for diagnostics.
    """

    last_decision: float = 0.0
    n_decisions: int = 0
    history: list[tuple[Observation, float]] | None = None


class Policy(ABC):
    """Abstract base class for treatment policies.

    Subclasses implement ``decide(obs)`` returning a drug level in [0, 1].
    The base class wraps it with state tracking and history logging.

    Conventions:
    - Drug levels are in [0, 1]. 0 = no drug. 1 = MTD.
    - The simulator runs forward continuously; the policy is queried at
      decision points (typically appointment intervals). Between decisions,
      the simulator holds the drug level constant at whatever the policy
      last returned.
    - Policies are deterministic by default. For stochastic policies, override
      and document.
    """

    name: str = "policy"
    """Display name for figures and reports. Override in subclasses."""

    def __init__(self, log_history: bool = False) -> None:
        self.state = PolicyState(history=[] if log_history else None)

    def __call__(self, obs: Observation) -> float:
        """Make a decision based on the observation."""
        decision = float(self.decide(obs))
        if not (0.0 <= decision <= 1.0):
            raise ValueError(
                f"Policy {self.name}: decision={decision} out of [0, 1]"
            )
        self.state.last_decision = decision
        self.state.n_decisions += 1
        if self.state.history is not None:
            self.state.history.append((obs, decision))
        return decision

    @abstractmethod
    def decide(self, obs: Observation) -> float:
        """Compute the drug level for this observation.

        Args:
            obs: Current observation.

        Returns:
            Drug level in [0, 1].
        """
        ...

    def reset(self) -> None:
        """Reset the policy to its initial state.

        Use between patients in a cohort run. Subclasses with extra state
        should override and call super().reset().
        """
        self.state = PolicyState(history=[] if self.state.history is not None else None)
