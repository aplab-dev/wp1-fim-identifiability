"""Tests for heuristic policies and the cohort runner.

Verifies:
- Each policy honors the [0, 1] action constraint.
- AT50 toggles correctly on PSA threshold crossings.
- Cohort runner produces a record per patient and computes summary stats.
"""

from __future__ import annotations

import numpy as np
import pytest

from policies.at50 import AT50Policy
from policies.base import Observation, Policy
from policies.cohort_runner import CohortResult, CohortRunner
from policies.mtd import MTDPolicy
from policies.no_treatment import NoTreatmentPolicy


# ---------- MTD ----------

class TestMTDPolicy:
    def test_always_returns_one(self) -> None:
        p = MTDPolicy()
        for t in [0.0, 50.0, 1000.0]:
            obs = Observation(t=t, psa=10.0, baseline_psa=20.0)
            assert p(obs) == 1.0

    def test_name(self) -> None:
        assert MTDPolicy().name == "MTD"


# ---------- No treatment ----------

class TestNoTreatmentPolicy:
    def test_always_returns_zero(self) -> None:
        p = NoTreatmentPolicy()
        for t in [0.0, 50.0, 1000.0]:
            obs = Observation(t=t, psa=10.0, baseline_psa=20.0)
            assert p(obs) == 0.0

    def test_name(self) -> None:
        assert NoTreatmentPolicy().name == "No treatment"


# ---------- AT50 ----------

class TestAT50Policy:
    def test_starts_with_drug_on(self) -> None:
        p = AT50Policy()
        # Initial observation, PSA at baseline -> still on
        obs = Observation(t=0.0, psa=20.0, baseline_psa=20.0)
        assert p(obs) == 1.0

    def test_withdraws_below_50pct(self) -> None:
        p = AT50Policy()
        # Start at baseline
        p(Observation(t=0.0, psa=20.0, baseline_psa=20.0))
        # Now drop below 50%
        result = p(Observation(t=10.0, psa=8.0, baseline_psa=20.0))  # 40% of baseline
        assert result == 0.0

    def test_does_not_withdraw_at_60pct(self) -> None:
        p = AT50Policy()
        p(Observation(t=0.0, psa=20.0, baseline_psa=20.0))
        # 60% of baseline — above the 50% withdrawal threshold
        result = p(Observation(t=10.0, psa=12.0, baseline_psa=20.0))
        assert result == 1.0

    def test_resumes_at_baseline(self) -> None:
        p = AT50Policy()
        # Drug on
        p(Observation(t=0.0, psa=20.0, baseline_psa=20.0))
        # Withdraw at 40%
        p(Observation(t=10.0, psa=8.0, baseline_psa=20.0))
        assert p.state.last_decision == 0.0
        # PSA back to baseline -> resume
        result = p(Observation(t=20.0, psa=20.0, baseline_psa=20.0))
        assert result == 1.0

    def test_does_not_resume_at_99pct(self) -> None:
        p = AT50Policy()
        p(Observation(t=0.0, psa=20.0, baseline_psa=20.0))
        p(Observation(t=10.0, psa=8.0, baseline_psa=20.0))  # withdraw
        # PSA at 99% of baseline — not yet at resumption threshold
        result = p(Observation(t=20.0, psa=19.8, baseline_psa=20.0))
        assert result == 0.0

    def test_full_cycle(self) -> None:
        """Drug on -> below 50% -> off -> at baseline -> on -> below 50% again."""
        p = AT50Policy()
        decisions = []
        # Phase 1: drug on, PSA dropping
        decisions.append(p(Observation(t=0, psa=20, baseline_psa=20)))   # on
        decisions.append(p(Observation(t=10, psa=15, baseline_psa=20)))  # on (75%)
        decisions.append(p(Observation(t=20, psa=8, baseline_psa=20)))   # off (40%)
        # Phase 2: drug off, PSA rising
        decisions.append(p(Observation(t=30, psa=12, baseline_psa=20)))  # off (60%)
        decisions.append(p(Observation(t=40, psa=20, baseline_psa=20)))  # on (100%)
        # Phase 3: drug on, PSA dropping again
        decisions.append(p(Observation(t=50, psa=10, baseline_psa=20)))  # off (50%)
        assert decisions == [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]

    def test_name_includes_threshold(self) -> None:
        assert AT50Policy().name == "AT50"
        assert AT50Policy(withdrawal_fraction=0.8).name == "AT80"

    def test_invalid_withdrawal_fraction(self) -> None:
        with pytest.raises(ValueError, match="withdrawal_fraction"):
            AT50Policy(withdrawal_fraction=1.5)

    def test_withdrawal_above_resumption_rejected(self) -> None:
        with pytest.raises(ValueError, match="withdrawal_fraction must be"):
            AT50Policy(withdrawal_fraction=0.8, resumption_fraction=0.7)

    def test_missing_psa_rejected(self) -> None:
        p = AT50Policy()
        with pytest.raises(ValueError, match="PSA"):
            p(Observation(t=0.0, psa=None, baseline_psa=20.0))

    def test_missing_baseline_rejected(self) -> None:
        p = AT50Policy()
        with pytest.raises(ValueError, match="baseline_psa"):
            p(Observation(t=0.0, psa=10.0, baseline_psa=None))

    def test_at80_variant(self) -> None:
        """AT80 uses 80% withdrawal threshold (Gallagher 2025)."""
        p = AT50Policy(withdrawal_fraction=0.8)
        p(Observation(t=0, psa=20, baseline_psa=20))
        # At 70% of baseline, AT80 should withdraw (below 0.8)
        assert p(Observation(t=10, psa=14, baseline_psa=20)) == 0.0


# ---------- Reset ----------

class TestPolicyReset:
    def test_reset_clears_state(self) -> None:
        p = AT50Policy(log_history=True)
        p(Observation(t=0.0, psa=20.0, baseline_psa=20.0))
        p(Observation(t=10.0, psa=8.0, baseline_psa=20.0))
        assert p.state.n_decisions == 2
        assert p.state.history is not None
        assert len(p.state.history) == 2
        p.reset()
        assert p.state.n_decisions == 0
        assert p.state.history is not None
        assert len(p.state.history) == 0


# ---------- Decision-range validation ----------

class _BadPolicy(Policy):
    name = "bad"

    def decide(self, obs: Observation) -> float:
        return 1.5  # out of [0, 1]


class TestPolicyDecisionRange:
    def test_out_of_range_rejected(self) -> None:
        p = _BadPolicy()
        obs = Observation(t=0.0, psa=10.0, baseline_psa=20.0)
        with pytest.raises(ValueError, match="out of"):
            p(obs)


# ---------- Cohort runner ----------

class TestCohortRunner:
    def test_runs_n_patients(self) -> None:
        """Cohort runner produces n_patients results."""

        def sampler(rng):
            return {"id": int(rng.integers(0, 1_000_000))}

        def run_one(params, policy, rng):
            return {
                "ttp": 100.0 + rng.normal(0, 10),
                "cumulative_dose": 0.5,
                "progressed": True,
                "patient_id": params["id"],
            }

        runner = CohortRunner(
            run_one_patient=run_one,
            param_sampler=sampler,
            n_patients=15,
            seed=42,
        )
        result = runner.run(policy_factory=MTDPolicy)
        assert isinstance(result, CohortResult)
        assert len(result.per_patient) == 15
        assert result.policy_name == "MTD"
        assert result.seed == 42

    def test_summary_stats(self) -> None:
        def sampler(rng):
            return {}

        def run_one(params, policy, rng):
            return {"ttp": 100.0, "cumulative_dose": 0.5, "progressed": True}

        runner = CohortRunner(
            run_one_patient=run_one,
            param_sampler=sampler,
            n_patients=5,
            seed=0,
        )
        result = runner.run(policy_factory=MTDPolicy)
        np.testing.assert_allclose(result.ttp_array(), [100.0] * 5)
        np.testing.assert_allclose(result.cumulative_dose_array(), [0.5] * 5)
        assert result.progression_rate() == 1.0

    def test_seed_reproducibility(self) -> None:
        """Same seed should produce identical patient parameters across runs."""

        def sampler(rng):
            return float(rng.normal())

        def run_one(params, policy, rng):
            return {"ttp": params * 100.0, "cumulative_dose": 0.0, "progressed": True}

        runner = CohortRunner(
            run_one_patient=run_one,
            param_sampler=sampler,
            n_patients=10,
            seed=12345,
        )
        result1 = runner.run(policy_factory=MTDPolicy)
        result2 = runner.run(policy_factory=MTDPolicy)
        np.testing.assert_array_equal(result1.ttp_array(), result2.ttp_array())

    def test_fresh_policy_per_patient(self) -> None:
        """Cohort runner creates a fresh policy per patient (state is reset)."""
        n_calls = []

        def sampler(rng):
            return {}

        def run_one(params, policy, rng):
            # Simulate calling the policy a few times
            policy(Observation(t=0.0, psa=10.0, baseline_psa=20.0))
            policy(Observation(t=10.0, psa=10.0, baseline_psa=20.0))
            n_calls.append(policy.state.n_decisions)
            return {"ttp": 100.0, "cumulative_dose": 0.0, "progressed": False}

        runner = CohortRunner(
            run_one_patient=run_one,
            param_sampler=sampler,
            n_patients=3,
            seed=0,
        )
        runner.run(policy_factory=MTDPolicy)
        # Each patient should see exactly 2 decisions, not cumulative
        assert n_calls == [2, 2, 2]
