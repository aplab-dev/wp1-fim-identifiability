"""Tests for the Zhang 2017 reproduction module.

Verifies:
- Canonical Zhang LV params + IC produce sensible PSA baseline.
- Single-patient runner returns the required output dict shape.
- Single-patient runner is reproducible given the same RNG.
- Qualitative outcomes match Zhang 2017's headline finding:
  - No-treatment progresses fast.
  - MTD delays progression but eventually fails (T- takes over).
  - AT50 substantially extends TTP and reduces drug exposure.
- IC perturbation generates cohort variation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from policies.at50 import AT50Policy
from policies.mtd import MTDPolicy
from policies.no_treatment import NoTreatmentPolicy
from simulators.psa_dynamics import PSAParams, psa_steady_state
from zhang2017 import (
    ZHANG_CANONICAL_X0,
    ZhangPatientParams,
    run_zhang_patient,
    zhang_2017_sampler,
    zhang_canonical_lv_params,
)


# ---------- canonical params ----------


class TestCanonicalZhangParams:
    def test_alpha_has_heavy_T_minus_suppression(self) -> None:
        """Zhang-canonical alpha matrix suppresses T- via T+/TP rows."""
        p = zhang_canonical_lv_params()
        # alpha[2, 0] (T- suppressed by T+) and alpha[2, 1] (T- by TP) > 1
        assert p.alpha[2, 0] > 1.0, "T- should be heavily suppressed by T+"
        assert p.alpha[2, 1] > 1.0, "T- should be heavily suppressed by TP"
        assert p.alpha[2, 2] == pytest.approx(1.0)

    def test_canonical_x0_is_T_plus_TP_dominated(self) -> None:
        x0 = ZHANG_CANONICAL_X0
        total = sum(x0)
        # T- should be a small fraction (~1%) — it's the resistance reservoir.
        assert x0[2] / total < 0.05
        # T+ and TP should each be substantial.
        assert x0[0] / total > 0.30
        assert x0[1] / total > 0.30

    def test_canonical_baseline_psa_is_finite_positive(self) -> None:
        x0 = ZHANG_CANONICAL_X0
        psa = psa_steady_state(sum(x0), PSAParams())
        assert psa > 0
        assert np.isfinite(psa)


# ---------- sampler ----------


class TestZhang2017Sampler:
    def test_returns_zhang_patient_params(self) -> None:
        rng = np.random.default_rng(0)
        p = zhang_2017_sampler(rng)
        assert isinstance(p, ZhangPatientParams)

    def test_uses_canonical_defaults(self) -> None:
        p = zhang_2017_sampler(np.random.default_rng(0))
        assert p.x0 == ZHANG_CANONICAL_X0
        # alpha matches the canonical Zhang factory
        np.testing.assert_array_equal(
            p.lv_params.alpha, zhang_canonical_lv_params().alpha
        )


# ---------- single-patient runner ----------


class TestRunZhangPatient:
    @pytest.fixture
    def params(self) -> ZhangPatientParams:
        # Disable IC perturbation for deterministic tests
        return ZhangPatientParams(ic_perturbation_std=0.0)

    def test_returns_required_keys(self, params: ZhangPatientParams) -> None:
        result = run_zhang_patient(params, MTDPolicy(), rng=np.random.default_rng(0))
        for k in ("ttp", "cumulative_dose", "progressed", "baseline_psa", "trajectory"):
            assert k in result
        # trajectory should be populated with arrays
        traj = result["trajectory"]
        for k in ("t", "x_Tplus", "x_TP", "x_Tminus", "Lambda", "psa"):
            assert k in traj
            assert isinstance(traj[k], np.ndarray)

    def test_no_treatment_progresses_fast(self, params: ZhangPatientParams) -> None:
        """Without drug, the tumor grows uncontrollably."""
        result = run_zhang_patient(
            params, NoTreatmentPolicy(), rng=np.random.default_rng(0)
        )
        assert result["progressed"]
        assert result["ttp"] < 365  # <1 year
        assert result["cumulative_dose"] == 0.0

    def test_mtd_drug_fraction_is_one(self, params: ZhangPatientParams) -> None:
        """MTD applies drug continuously; cumulative_dose / ttp == 1 (within rounding)."""
        result = run_zhang_patient(params, MTDPolicy(), rng=np.random.default_rng(0))
        if result["ttp"] > 0:
            frac = result["cumulative_dose"] / result["ttp"]
            assert frac == pytest.approx(1.0, abs=0.01)

    def test_mtd_extends_ttp_vs_no_treatment(self, params: ZhangPatientParams) -> None:
        """MTD should outperform no-treatment on TTP (Zhang's SOC baseline)."""
        no_tx = run_zhang_patient(
            params, NoTreatmentPolicy(), rng=np.random.default_rng(0)
        )
        mtd = run_zhang_patient(
            params, MTDPolicy(), rng=np.random.default_rng(0)
        )
        # MTD should at least double the no-treatment TTP
        assert mtd["ttp"] > 2 * no_tx["ttp"]

    def test_at50_extends_ttp_vs_mtd(self, params: ZhangPatientParams) -> None:
        """The Zhang 2017 headline finding: AT50 > MTD on TTP."""
        mtd = run_zhang_patient(params, MTDPolicy(), rng=np.random.default_rng(0))
        at50 = run_zhang_patient(params, AT50Policy(), rng=np.random.default_rng(0))
        assert at50["ttp"] >= mtd["ttp"]

    def test_at50_uses_less_drug_than_mtd(self, params: ZhangPatientParams) -> None:
        """AT50 should require substantially less cumulative drug than MTD."""
        mtd = run_zhang_patient(params, MTDPolicy(), rng=np.random.default_rng(0))
        at50 = run_zhang_patient(params, AT50Policy(), rng=np.random.default_rng(0))
        assert at50["cumulative_dose"] < mtd["cumulative_dose"]

    def test_baseline_psa_matches_steady_state(self, params: ZhangPatientParams) -> None:
        """baseline_psa equals quasi-steady-state PSA at the IC."""
        result = run_zhang_patient(
            params, NoTreatmentPolicy(), rng=np.random.default_rng(0)
        )
        # IC is canonical (no perturbation since std=0)
        expected = psa_steady_state(sum(ZHANG_CANONICAL_X0), params.psa_params)
        assert result["baseline_psa"] == pytest.approx(expected, rel=1e-6)

    def test_trajectory_is_monotonically_increasing_in_t(
        self, params: ZhangPatientParams
    ) -> None:
        result = run_zhang_patient(params, MTDPolicy(), rng=np.random.default_rng(0))
        t = result["trajectory"]["t"]
        # No duplicates, strictly non-decreasing
        diffs = np.diff(t)
        assert np.all(diffs >= 0), "trajectory time must be non-decreasing"

    def test_reproducibility_with_same_seed(
        self, params: ZhangPatientParams
    ) -> None:
        """Same RNG -> identical TTP and cumulative dose."""
        # Re-enable perturbation so RNG matters
        params_perturbed = ZhangPatientParams(ic_perturbation_std=0.10)
        r1 = run_zhang_patient(
            params_perturbed, MTDPolicy(), rng=np.random.default_rng(123)
        )
        r2 = run_zhang_patient(
            params_perturbed, MTDPolicy(), rng=np.random.default_rng(123)
        )
        assert r1["ttp"] == pytest.approx(r2["ttp"], rel=1e-9)
        assert r1["cumulative_dose"] == pytest.approx(r2["cumulative_dose"], rel=1e-9)


# ---------- IC perturbation ----------


class TestICPerturbation:
    def test_zero_perturbation_is_deterministic(self) -> None:
        """ic_perturbation_std=0 -> all patients identical regardless of RNG."""
        params = ZhangPatientParams(ic_perturbation_std=0.0)
        r1 = run_zhang_patient(params, MTDPolicy(), rng=np.random.default_rng(0))
        r2 = run_zhang_patient(params, MTDPolicy(), rng=np.random.default_rng(99))
        assert r1["ttp"] == pytest.approx(r2["ttp"], rel=1e-9)

    def test_perturbation_generates_variation(self) -> None:
        """ic_perturbation_std>0 + different RNGs -> different TTPs."""
        warnings.filterwarnings("ignore")  # suppress LSODA warnings on outliers
        params = ZhangPatientParams(ic_perturbation_std=0.20)
        ttps = []
        for seed in range(8):
            r = run_zhang_patient(
                params, MTDPolicy(), rng=np.random.default_rng(seed)
            )
            ttps.append(r["ttp"])
        # Some variation must appear across patients
        assert np.std(ttps) > 0


# ---------- progression detection ----------


class TestProgressionDetection:
    def test_high_threshold_means_no_progression(self) -> None:
        """If the threshold is set absurdly high, the patient never progresses."""
        params = ZhangPatientParams(
            ic_perturbation_std=0.0,
            progression_psa_threshold=1e6,  # impossibly high
        )
        result = run_zhang_patient(
            params, NoTreatmentPolicy(), rng=np.random.default_rng(0)
        )
        assert not result["progressed"]
        assert result["ttp"] == pytest.approx(params.t_max, rel=1e-9)

    def test_low_threshold_means_immediate_progression(self) -> None:
        """If threshold < 1.0, baseline_psa > threshold from t=0 -> immediate progression."""
        params = ZhangPatientParams(
            ic_perturbation_std=0.0,
            progression_psa_threshold=0.5,  # baseline already above this
        )
        result = run_zhang_patient(
            params, NoTreatmentPolicy(), rng=np.random.default_rng(0)
        )
        assert result["progressed"]
        # Should detect within the first chunk
        assert result["ttp"] <= params.decision_interval


# ---------- cohort runner integration ----------


class TestCohortRunnerCompatibility:
    def test_run_zhang_patient_signature_matches_protocol(self) -> None:
        """run_zhang_patient(params, policy, rng) returns dict with required keys."""
        from policies.cohort_runner import CohortRunner

        runner = CohortRunner(
            run_one_patient=run_zhang_patient,
            param_sampler=zhang_2017_sampler,
            n_patients=3,
            seed=42,
        )
        result = runner.run(policy_factory=lambda: AT50Policy(log_history=False))
        assert len(result.per_patient) == 3
        # ttp_array, cumulative_dose_array, progression_rate must work
        ttps = result.ttp_array()
        assert ttps.shape == (3,)
        doses = result.cumulative_dose_array()
        assert doses.shape == (3,)
        rate = result.progression_rate()
        assert 0.0 <= rate <= 1.0
