"""Cohort runner — apply a Policy to a (simulator, parameter-distribution) pair.

Decoupled from any specific simulator: pass in a callable ``run_one_patient``
that takes (params, policy, ...) and returns a record of outcomes (TTP,
cumulative dose, etc.). This abstracts over 2-pop vs 3-pop simulator details.

Usage pattern:

    runner = CohortRunner(
        run_one_patient=run_zhang_patient,
        param_sampler=zhang_2017_sampler,
        n_patients=200,
        seed=0,
    )
    results = runner.run(MTDPolicy())

Returns a list of patient-result dicts, ready for analysis.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .base import Policy

log = logging.getLogger(__name__)


class ParamSampler(Protocol):
    """Callable that samples a parameter set from a patient distribution."""

    def __call__(self, rng: np.random.Generator) -> Any:
        """Return a parameter object suitable for ``run_one_patient``."""
        ...


class RunOnePatient(Protocol):
    """Callable that simulates one patient under a policy.

    Args:
        params: Parameter object (e.g., LV2PopParams or LV3PopParams).
        policy: Policy instance to apply.
        rng: Optional RNG for stochastic experiments.

    Returns:
        Dict with at least:
        - ``ttp``: time to progression (days). float.
        - ``cumulative_dose``: integral of u(t) over the run. float.
        - ``progressed``: bool — did the patient cross the progression threshold.
        Plus any other diagnostic fields the simulator chooses to emit.
    """

    def __call__(
        self,
        params: Any,
        policy: Policy,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class CohortResult:
    """Outcome of running a policy over a cohort.

    Attributes:
        policy_name: Display name of the policy applied.
        per_patient: List of per-patient result dicts (from run_one_patient).
        seed: RNG seed used to sample patients.
    """

    policy_name: str
    per_patient: list[dict[str, Any]] = field(default_factory=list)
    seed: int = 0

    def ttp_array(self) -> np.ndarray:
        """Vector of time-to-progression values across the cohort."""
        return np.array([r["ttp"] for r in self.per_patient])

    def cumulative_dose_array(self) -> np.ndarray:
        """Vector of cumulative-dose values across the cohort."""
        return np.array([r["cumulative_dose"] for r in self.per_patient])

    def progression_rate(self) -> float:
        """Fraction of patients who progressed within the simulation horizon."""
        return float(np.mean([r["progressed"] for r in self.per_patient]))


@dataclass
class CohortRunner:
    """Runs a Policy across a sampled patient cohort.

    Attributes:
        run_one_patient: Callable that simulates one patient under a policy.
        param_sampler: Callable that samples one patient's parameters.
        n_patients: Number of patients to simulate.
        seed: Top-level seed; per-patient seeds are derived deterministically.
    """

    run_one_patient: RunOnePatient
    param_sampler: ParamSampler
    n_patients: int = 200
    seed: int = 0

    def run(
        self,
        policy_factory: Callable[[], Policy],
        verbose: bool = False,
    ) -> CohortResult:
        """Run the policy across the cohort.

        Args:
            policy_factory: Callable that returns a fresh Policy instance.
                We need to instantiate fresh per-patient because policies
                are stateful (AT50 tracks drug-on/off phase).
            verbose: If True, log progress every 20 patients.

        Returns:
            CohortResult with one entry per simulated patient.
        """
        # Use SeedSequence to get reproducible per-patient streams.
        ss = np.random.SeedSequence(self.seed)
        child_seeds = ss.spawn(self.n_patients)

        # Instantiate one policy to read its name (for the result label).
        sample_policy = policy_factory()
        policy_name = sample_policy.name

        per_patient: list[dict[str, Any]] = []
        for i, child in enumerate(child_seeds):
            rng = np.random.Generator(np.random.PCG64(child))
            params = self.param_sampler(rng)
            policy = policy_factory()  # fresh state per patient
            result = self.run_one_patient(params, policy, rng)
            per_patient.append(result)
            if verbose and (i + 1) % 20 == 0:
                log.info(f"  cohort progress: {i + 1}/{self.n_patients}")

        return CohortResult(
            policy_name=policy_name,
            per_patient=per_patient,
            seed=self.seed,
        )
