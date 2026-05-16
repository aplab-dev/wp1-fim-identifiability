"""Zhang 2017 AT50 protocol — drug on/off based on PSA threshold.

Per Zhang 2017 reassessment §"Correction 5":

- **Initiation:** Abiraterone (drug ON) until PSA achieves >=50% decline
  from pre-treatment baseline.
- **Withdrawal:** Drug OFF when PSA crosses below the 50%-of-baseline
  threshold from above.
- **Resumption:** Drug ON when PSA returns to baseline (100% of pre-treatment
  value).
- **Repeat:** cycle indefinitely until clinical/radiographic progression.

This is a stateful policy: it tracks whether it's currently in the "on"
or "off" phase, and switches based on PSA crossings.

Note: real Zhang 2017 patients are observed at 4-week labs and 12-week
imaging intervals. The policy here decides at *every* time the simulator
queries it (which is at the simulator's adaptive ODE-step granularity by
default, or at any custom decision-cadence the experiment specifies).
For more clinically realistic behavior, wrap in a periodic-decision
sampler at the `cohort_runner` level.

For an AT-X variant (e.g., AT80 from Gallagher 2025), pass `withdrawal_fraction`.
For Zhang's exact protocol, the default 0.5 reproduces AT50.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import Observation, Policy, PolicyState


@dataclass
class AT50State(PolicyState):
    """State for AT50: track current on/off phase and the threshold."""

    in_drug_phase: bool = True
    """Whether drug is currently ON. Starts True (per Zhang 2017 protocol)."""

    has_reached_first_threshold: bool = False
    """Has the patient ever reached the X%-decline threshold? Until they have,
    the protocol stays in the initial drug-on phase even if PSA fluctuates."""


class AT50Policy(Policy):
    """Zhang 2017 adaptive protocol with PSA-threshold switching.

    Args:
        withdrawal_fraction: Fraction of baseline PSA at which to withdraw
            drug (Zhang 2017 = 0.5; Gallagher 2025 AT80 = 0.8).
        resumption_fraction: Fraction of baseline at which to resume drug
            (Zhang 2017 = 1.0). Setting > 1.0 allows tumor to overshoot
            baseline before re-treating.
    """

    name: str = "AT50"

    def __init__(
        self,
        withdrawal_fraction: float = 0.5,
        resumption_fraction: float = 1.0,
        log_history: bool = False,
    ):
        if not 0.0 <= withdrawal_fraction <= 1.0:
            raise ValueError(
                f"withdrawal_fraction must be in [0, 1]; got {withdrawal_fraction}"
            )
        if resumption_fraction <= 0.0:
            raise ValueError(
                f"resumption_fraction must be positive; got {resumption_fraction}"
            )
        if withdrawal_fraction >= resumption_fraction:
            raise ValueError(
                "withdrawal_fraction must be < resumption_fraction; "
                f"got {withdrawal_fraction} >= {resumption_fraction}"
            )
        super().__init__(log_history=log_history)
        # Initialize AT50-specific state. We can't use super's PolicyState
        # because we need extra fields; subclass it.
        self.state = AT50State(history=[] if log_history else None)
        self.withdrawal_fraction = withdrawal_fraction
        self.resumption_fraction = resumption_fraction
        self.name = f"AT{int(withdrawal_fraction * 100):d}"

    def decide(self, obs: Observation) -> float:
        if obs.psa is None:
            raise ValueError(
                f"{self.name} policy requires PSA in the observation; got None"
            )
        if obs.baseline_psa is None or obs.baseline_psa <= 0:
            raise ValueError(
                f"{self.name} policy requires baseline_psa > 0; got {obs.baseline_psa}"
            )

        s: AT50State = self.state  # type: ignore[assignment]
        psa_ratio = obs.psa / obs.baseline_psa

        # Check threshold crossings
        if s.in_drug_phase:
            # Drug is on. Should we withdraw?
            if psa_ratio <= self.withdrawal_fraction:
                # Crossed below the withdrawal threshold from above.
                s.in_drug_phase = False
                s.has_reached_first_threshold = True
        else:
            # Drug is off. Should we resume?
            if psa_ratio >= self.resumption_fraction:
                s.in_drug_phase = True

        return 1.0 if s.in_drug_phase else 0.0

    def reset(self) -> None:
        super().reset()
        self.state = AT50State(history=self.state.history)
