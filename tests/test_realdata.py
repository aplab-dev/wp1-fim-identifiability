"""Tests for the realdata module — Bruchovsky cohort schema + per-patient MCMC.

Verifies:
- BruchovskyPatient validation (shapes, monotone time, non-negative PSA).
- generate_synthetic_cohort returns a cohort of the right size + structure.
- load_cohort_csv round-trips.
- rhat_split has correct shape and reduces to ~1.0 for equal-distribution chains.
- fit_patient_mcmc runs without error on a synthetic patient with reasonable n_steps.
- MCMCResult.flat_samples and converged() work as expected.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pytest

from realdata import (
    BruchovskyCohort,
    BruchovskyPatient,
    MCMCResult,
    fit_patient_mcmc,
    generate_synthetic_cohort,
    load_cohort_csv,
    load_dataTanaka,
    rhat_split,
)


@pytest.fixture(autouse=True)
def _suppress_warnings():
    warnings.filterwarnings("ignore")
    yield


# ---------- BruchovskyPatient ----------


class TestBruchovskyPatient:
    def test_basic_construction(self) -> None:
        t = np.array([0, 28, 56, 84.0])
        psa = np.array([10, 5, 2, 8.0])
        u = np.array([1, 1, 0, 0.0])
        p = BruchovskyPatient(patient_id="test", t_obs=t, psa_obs=psa, u_schedule=u)
        assert p.n_obs() == 4
        assert p.baseline == 10.0  # baseline_psa not given => uses psa_obs[0]

    def test_explicit_baseline(self) -> None:
        t = np.array([0, 28.0])
        psa = np.array([5, 6.0])
        u = np.array([1, 0.0])
        p = BruchovskyPatient(patient_id="x", t_obs=t, psa_obs=psa, u_schedule=u, baseline_psa=42.0)
        assert p.baseline == 42.0

    def test_shape_mismatch_rejected(self) -> None:
        t = np.array([0, 28.0])
        psa = np.array([5.0])
        u = np.array([1.0, 0.0])
        with pytest.raises(ValueError):
            BruchovskyPatient(patient_id="bad", t_obs=t, psa_obs=psa, u_schedule=u)

    def test_negative_psa_rejected(self) -> None:
        t = np.array([0, 28.0])
        psa = np.array([5, -1.0])
        u = np.array([1, 0.0])
        with pytest.raises(ValueError):
            BruchovskyPatient(patient_id="bad", t_obs=t, psa_obs=psa, u_schedule=u)

    def test_non_increasing_time_rejected(self) -> None:
        t = np.array([0, 28, 14.0])  # not monotone
        psa = np.array([5, 6, 7.0])
        u = np.array([1, 0, 0.0])
        with pytest.raises(ValueError):
            BruchovskyPatient(patient_id="bad", t_obs=t, psa_obs=psa, u_schedule=u)


# ---------- generate_synthetic_cohort ----------


class TestGenerateSyntheticCohort:
    def test_returns_cohort(self) -> None:
        cohort = generate_synthetic_cohort(n_patients=5, seed=42)
        assert isinstance(cohort, BruchovskyCohort)
        assert cohort.source == "synthetic"

    def test_size_matches_request(self) -> None:
        # In rare cases a patient may fail to integrate; allow at least 80% success.
        cohort = generate_synthetic_cohort(n_patients=10, seed=0)
        assert cohort.n_patients >= 8

    def test_patients_have_ids(self) -> None:
        cohort = generate_synthetic_cohort(n_patients=3, seed=0)
        ids = [p.patient_id for p in cohort.patients]
        assert len(ids) == len(set(ids))  # unique

    def test_progression_rate_computable(self) -> None:
        cohort = generate_synthetic_cohort(n_patients=10, seed=0)
        rate = cohort.progression_rate()
        assert 0.0 <= rate <= 1.0

    def test_seed_reproducibility(self) -> None:
        c1 = generate_synthetic_cohort(n_patients=5, seed=123)
        c2 = generate_synthetic_cohort(n_patients=5, seed=123)
        # PSA arrays should match
        for p1, p2 in zip(c1.patients, c2.patients):
            np.testing.assert_allclose(p1.psa_obs, p2.psa_obs, rtol=1e-9)


# ---------- load_cohort_csv ----------


class TestLoadCohortCsv:
    def test_roundtrip(self, tmp_path) -> None:
        csv_path = tmp_path / "test_cohort.csv"
        with csv_path.open("w") as f:
            w = csv.writer(f)
            w.writerow(["patient_id", "t_obs_days", "psa", "u_schedule",
                        "baseline_psa", "progression_observed", "ttp_observed"])
            for t, psa, u in [(0.0, 10.0, 1), (28.0, 7.0, 1), (56.0, 4.0, 0)]:
                w.writerow(["p001", t, psa, u, 10.0, 0, ""])
            for t, psa, u in [(0.0, 8.0, 1), (28.0, 6.0, 1), (56.0, 12.0, 0)]:
                w.writerow(["p002", t, psa, u, 8.0, 1, 56])
        cohort = load_cohort_csv(csv_path)
        assert cohort.n_patients == 2
        ids = {p.patient_id for p in cohort.patients}
        assert ids == {"p001", "p002"}
        p2 = next(p for p in cohort.patients if p.patient_id == "p002")
        assert p2.progression_observed
        assert p2.ttp_observed == 56.0

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_cohort_csv(tmp_path / "does_not_exist.csv")


# ---------- load_dataTanaka (real Bruchovsky cohort) ----------


class TestLoadDataTanaka:
    @pytest.fixture(scope="class")
    def real_cohort(self):
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[1]
        data_root = repo_root / "data" / "raw" / "dataTanaka" / "Bruchovsky_et_al"
        if not data_root.exists():
            pytest.skip("Real cohort data not present in data/raw/dataTanaka")
        return load_dataTanaka()

    def test_loads_real_cohort(self, real_cohort) -> None:
        assert real_cohort.n_patients > 50  # expect ~72
        assert "Bruchovsky_et_al" in real_cohort.source

    def test_real_patients_have_psa_baseline(self, real_cohort) -> None:
        for p in real_cohort.patients[:10]:
            assert p.baseline > 0
            assert p.n_obs() >= 10

    def test_real_progression_rate_in_range(self, real_cohort) -> None:
        rate = real_cohort.progression_rate()
        # Brady-Nicholls 2020 reports a substantial fraction progressed
        assert 0.20 < rate < 0.80

    def test_real_patient_ids_unique(self, real_cohort) -> None:
        ids = [p.patient_id for p in real_cohort.patients]
        assert len(ids) == len(set(ids))


class TestLoadShawEtAl:
    @pytest.fixture(scope="class")
    def shaw_cohort(self):
        from pathlib import Path
        from realdata import load_shaw_et_al
        repo_root = Path(__file__).resolve().parents[1]
        data_root = repo_root / "data" / "raw" / "dataTanaka" / "Shaw_et_al"
        if not data_root.exists():
            pytest.skip("Shaw cohort data not present")
        return load_shaw_et_al()

    def test_loads_shaw_cohort(self, shaw_cohort) -> None:
        # 18 files in archive, ≥10 obs filter may drop a few — expect ≥15
        assert shaw_cohort.n_patients >= 15
        assert "Shaw_et_al" in shaw_cohort.source

    def test_shaw_progression_rate_in_range(self, shaw_cohort) -> None:
        rate = shaw_cohort.progression_rate()
        assert 0.20 < rate < 0.95


# ---------- rhat_split ----------


class TestRhatSplit:
    def test_iid_chains_close_to_one(self) -> None:
        # 4 chains of 200 samples from N(0, 1) — expect rhat ~ 1.0
        rng = np.random.default_rng(0)
        s = rng.normal(size=(4, 200, 3))
        rhat = rhat_split(s)
        # All R-hats should be < 1.10 for IID gaussian
        assert np.all(rhat < 1.10), f"R-hat values: {rhat}"

    def test_divergent_chains_high_rhat(self) -> None:
        # 4 chains with widely different means -> high R-hat
        rng = np.random.default_rng(1)
        s = np.empty((4, 100, 2))
        for i in range(4):
            s[i] = rng.normal(loc=10 * i, size=(100, 2))
        rhat = rhat_split(s)
        # At least one R-hat should be high
        assert np.any(rhat > 2.0)

    def test_too_few_chains_returns_inf(self) -> None:
        s = np.zeros((1, 100, 3))
        rhat = rhat_split(s)
        assert np.all(np.isinf(rhat))

    def test_returns_correct_shape(self) -> None:
        s = np.zeros((3, 50, 6))
        rhat = rhat_split(s)
        assert rhat.shape == (6,)


# ---------- fit_patient_mcmc ----------


class TestFitPatientMcmc:
    @pytest.fixture
    def synthetic_patient(self) -> BruchovskyPatient:
        cohort = generate_synthetic_cohort(n_patients=1, seed=0)
        return cohort.patients[0]

    def test_returns_mcmc_result(self, synthetic_patient) -> None:
        result = fit_patient_mcmc(
            synthetic_patient, n_chains=2, n_steps=400, burn_in=150, thin=4, seed=0,
        )
        assert isinstance(result, MCMCResult)

    def test_samples_shape(self, synthetic_patient) -> None:
        result = fit_patient_mcmc(
            synthetic_patient, n_chains=2, n_steps=400, burn_in=150, thin=4, seed=0,
        )
        assert result.samples.shape[0] == 2  # n_chains
        assert result.samples.shape[2] == 6  # n_params

    def test_rhat_shape(self, synthetic_patient) -> None:
        result = fit_patient_mcmc(
            synthetic_patient, n_chains=2, n_steps=400, burn_in=150, thin=4, seed=0,
        )
        assert result.rhat.shape == (6,)

    def test_flat_samples(self, synthetic_patient) -> None:
        result = fit_patient_mcmc(
            synthetic_patient, n_chains=2, n_steps=400, burn_in=150, thin=4, seed=0,
        )
        flat = result.flat_samples()
        assert flat.shape[1] == 6
        assert flat.shape[0] == result.n_chains * result.n_samples_per_chain

    def test_converged_method(self, synthetic_patient) -> None:
        result = fit_patient_mcmc(
            synthetic_patient, n_chains=2, n_steps=400, burn_in=150, thin=4, seed=0,
        )
        # Whether actually converged at 400 steps is a separate question — we just
        # check the method runs and returns a bool.
        c = result.converged(rhat_threshold=1.5)
        assert isinstance(c, bool)
