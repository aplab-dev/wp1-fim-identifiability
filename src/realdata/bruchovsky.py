"""Bruchovsky 2008 IADT cohort schema + synthetic-cohort generator.

The Bruchovsky 2008 Phase II intermittent androgen deprivation trial
(n=109 enrolled, n=70 with usable PSA series per Brady-Nicholls 2020 /
n=65 per Strobl 2022 inclusion criteria) is the most-fit cohort in the
adaptive-cancer-therapy literature. PSA measurements typically span
several intermittent treatment cycles per patient, with measurement
cadence ~28 days and patient follow-up of 5+ years.

Real-data ingestion path (when supplementary data is acquired):
1. Pull supplementary tables from Bruchovsky 2008, Brady-Nicholls 2020, or
   archived Vancouver-Coastal-Health datasets via author request.
2. Convert to the ``BruchovskyPatient`` dataclass schema below.
3. Pass the resulting ``BruchovskyCohort`` to the per-patient MCMC layer.

Until real data arrives, this module's ``generate_synthetic_cohort()``
produces a synthetic cohort with the same structural shape, drawn from
the FIM-induced posterior on the 3-pop K-shift Zhang model. This lets
us validate the entire M7 + M8 pipeline end-to-end before any real-data
acquisition gates.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BruchovskyPatient:
    """One patient's PSA trajectory + metadata.

    Attributes:
        patient_id: opaque identifier (string-typed for compatibility with
            real Bruchovsky IDs which can be alphanumeric).
        t_obs: (N,) measurement times in days from treatment initiation.
        psa_obs: (N,) PSA values at those times.
        u_schedule: (N,) reported drug schedule, 0/1 binary, at the
            measurement times. Real Bruchovsky data has explicit per-cycle
            on/off windows; we discretize to per-measurement.
        baseline_psa: pre-treatment baseline PSA used for AT-style
            threshold decisions. If None at construction time, computed as
            ``psa_obs[0]``.
        progression_observed: bool — did the patient progress within follow-up.
        ttp_observed: time-to-progression in days, or None if right-censored.
        notes: free-text annotations (cycle structure, dose adjustments).
    """

    patient_id: str
    t_obs: np.ndarray
    psa_obs: np.ndarray
    u_schedule: np.ndarray
    baseline_psa: float | None = None
    progression_observed: bool = False
    ttp_observed: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.t_obs.shape != self.psa_obs.shape:
            raise ValueError(
                f"t_obs and psa_obs must match in shape; got {self.t_obs.shape} vs {self.psa_obs.shape}"
            )
        if self.u_schedule.shape != self.t_obs.shape:
            raise ValueError(
                f"u_schedule must match t_obs shape; got {self.u_schedule.shape} vs {self.t_obs.shape}"
            )
        if np.any(self.psa_obs < 0):
            raise ValueError("psa_obs must be non-negative")
        if not np.all(np.diff(self.t_obs) > 0):
            raise ValueError("t_obs must be strictly increasing")

    @property
    def baseline(self) -> float:
        return self.baseline_psa if self.baseline_psa is not None else float(self.psa_obs[0])

    def n_obs(self) -> int:
        return self.t_obs.size


@dataclass(frozen=True)
class BruchovskyCohort:
    """Container for an n-patient cohort.

    Attributes:
        patients: list of BruchovskyPatient.
        source: 'synthetic', 'bruchovsky2008', 'brady_nicholls_2020',
            'zhang2017', or other.
        date_acquired: ISO date string when the data was loaded.
        n_patients: convenience property.
    """

    patients: list[BruchovskyPatient]
    source: str = "synthetic"
    date_acquired: str = ""

    @property
    def n_patients(self) -> int:
        return len(self.patients)

    def progression_rate(self) -> float:
        return float(np.mean([p.progression_observed for p in self.patients]))

    def median_ttp(self) -> float:
        ttps = [p.ttp_observed for p in self.patients if p.ttp_observed is not None]
        if not ttps:
            return float("nan")
        return float(np.median(ttps))


def _zhang_canonical_theta() -> np.ndarray:
    """Canonical Zhang theta — used as the population-mean for synthetic cohort."""
    from zhang2017 import zhang_canonical_lv_params  # local import: avoid circular at module load
    canon = zhang_canonical_lv_params()
    return np.array([
        canon.r_Tplus, canon.r_TP, canon.r_Tminus,
        float(canon.alpha[2, 0]), float(canon.alpha[2, 1]),
        canon.K_TP_drop,
    ])


def _simulate_one_patient_psa(
    theta: np.ndarray,
    u_schedule: np.ndarray,
    t_obs: np.ndarray,
    rng: np.random.Generator,
    psa_noise_rel: float = 0.10,
) -> np.ndarray:
    """Simulate one patient's PSA at t_obs given (theta, schedule, RNG)."""
    from scipy.integrate import solve_ivp

    from simulators.lv_3pop_kshift import LV3PopKShift, LV3PopParams
    from simulators.psa_dynamics import PSAParams, psa_steady_state
    from zhang2017 import ZHANG_CANONICAL_X0, zhang_canonical_lv_params

    canon = zhang_canonical_lv_params()
    alpha = canon.alpha.copy()
    alpha[2, 0] = max(theta[3], 0.01)
    alpha[2, 1] = max(theta[4], 0.01)
    params = LV3PopParams(
        r_Tplus=max(theta[0], 1e-6), r_TP=max(theta[1], 1e-6), r_Tminus=max(theta[2], 1e-6),
        K_Tminus=canon.K_Tminus, K_TP_max=canon.K_TP_max,
        K_TP_drop=max(min(theta[5], canon.K_TP_max - 1), 1.0),
        mu_max=canon.mu_max, mu_drop=canon.mu_drop,
        alpha=alpha,
    )
    sim = LV3PopKShift(params)
    psa_params = PSAParams()

    def rhs(t, y):
        x = y[:3]
        psa = y[3]
        # piecewise-constant u from schedule
        idx = int(np.searchsorted(t_obs, t, side="right") - 1)
        idx = max(0, min(idx, len(u_schedule) - 1))
        u = float(u_schedule[idx])
        dx = sim.dynamics(t, x, u)
        dpsa = psa_params.rho * float(np.sum(x)) - psa_params.phi * psa
        return np.concatenate([dx, [dpsa]])

    psa0 = psa_steady_state(float(np.sum(ZHANG_CANONICAL_X0)), psa_params)
    y0 = np.array([*ZHANG_CANONICAL_X0, psa0])
    for method in ("LSODA", "BDF"):
        try:
            sol = solve_ivp(rhs, t_span=(t_obs[0], t_obs[-1]), y0=y0, t_eval=t_obs,
                            method=method, rtol=1e-7, atol=1e-3)
            if sol.success:
                psa_clean = sol.y[3]
                noise = rng.normal(size=psa_clean.shape) * psa_noise_rel * np.maximum(psa_clean, 0.1 * psa_clean.max())
                return np.maximum(psa_clean + noise, 0.0)
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("synthetic patient PSA simulation failed under both LSODA and BDF")


def generate_synthetic_cohort(
    n_patients: int = 70,
    seed: int = 0,
    t_max: float = 1500.0,
    measurement_interval: float = 28.0,
    cycle_length: float = 280.0,
    cycle_duty: float = 0.5,
    inter_patient_log_std: float = 0.15,
) -> BruchovskyCohort:
    """Generate a Bruchovsky-2008-shaped synthetic cohort.

    Each patient gets:
    - A per-patient theta drawn log-normally around the Zhang-canonical mean
      with std `inter_patient_log_std` per parameter (15% by default).
    - A periodic-cycle drug schedule (cycle_length days, duty cycle =
      cycle_duty) — mimics the actual Bruchovsky IADT pattern.
    - PSA observations at `measurement_interval`-day cadence over `t_max`.
    - Progression flagged if PSA exceeds 1.20 × baseline at any point.

    Args:
        n_patients: cohort size (default 70 to match Bruchovsky n_usable).
        seed: top-level RNG seed for reproducibility.
        t_max: follow-up horizon in days (default 1500 ≈ 50 months).
        measurement_interval: PSA-lab cadence in days (default 28).
        cycle_length: IADT cycle length in days (default 280 ≈ 9 months).
        cycle_duty: fraction of each cycle on drug (default 0.5).
        inter_patient_log_std: log-normal SD for per-patient theta perturbation.

    Returns:
        BruchovskyCohort with `n_patients` synthetic patients.
    """
    rng_master = np.random.default_rng(seed)
    theta_canon = _zhang_canonical_theta()
    t_obs = np.arange(0.0, t_max + 1, measurement_interval)
    # Periodic drug schedule
    u_schedule = np.array([1.0 if (t % cycle_length) < cycle_duty * cycle_length else 0.0
                           for t in t_obs])

    patients: list[BruchovskyPatient] = []
    for i in range(n_patients):
        patient_seed = int(rng_master.integers(0, 1_000_000))
        patient_rng = np.random.default_rng(patient_seed)
        log_pert = patient_rng.normal(0, inter_patient_log_std, size=6)
        theta_i = theta_canon * np.exp(log_pert)
        # Clamp K_TP_drop within bounds
        theta_i[5] = np.clip(theta_i[5], 100.0, 9999.0)
        try:
            psa_clean = _simulate_one_patient_psa(theta_i, u_schedule, t_obs, patient_rng)
        except RuntimeError:
            continue  # skip if integration fails for this draw
        baseline = float(psa_clean[0])
        progression_thresh = 1.20 * baseline
        prog_idx = np.where(psa_clean >= progression_thresh)[0]
        if prog_idx.size > 0 and prog_idx[0] > 0:
            ttp = float(t_obs[prog_idx[0]])
            progressed = True
        else:
            ttp = None
            progressed = False
        patients.append(BruchovskyPatient(
            patient_id=f"synth_{i:03d}",
            t_obs=t_obs.copy(),
            psa_obs=psa_clean,
            u_schedule=u_schedule.copy(),
            baseline_psa=baseline,
            progression_observed=progressed,
            ttp_observed=ttp,
            notes=f"synthetic; per-patient log_pert={log_pert.tolist()}",
        ))

    import datetime as dt
    return BruchovskyCohort(
        patients=patients,
        source="synthetic",
        date_acquired=dt.date.today().isoformat(),
    )


def load_dataTanaka(
    data_root: Path | str = "data/raw/dataTanaka/Bruchovsky_et_al",
    classifications_path: Path | str | None = None,
    min_obs: int = 10,
    progression_psa_factor: float = 1.20,
) -> BruchovskyCohort:
    """Load the Bruchovsky et al. real-patient cohort from dataTanaka.zip.

    Source: http://www.nicholasbruchovsky.com/dataTanaka.zip — public repository
    accompanying Tanaka et al.'s mathematical-modeling paper. Same cohort
    Brady-Nicholls 2020, Strobl 2022, and Gallagher 2025 fit. ~72 patient
    files; not all have full follow-up, so we filter to patients with
    ``min_obs`` (default 10) PSA observations.

    File format (per readme.txt):
        col 1: patient_id (int)
        col 2: date (YYYY/M/D HH:MM:SS)
        col 3: CPA dose (mg, may be quoted)
        col 4: LEU dose (mg, may be quoted)
        col 5: PSA (ng/mL)
        col 6: testosterone (nmol/L)
        col 7: cycle number
        col 8: treatment (1=on, 0=off) — used as u_schedule
        col 9: day number (relative)
        col 10: day number (alternate origin) — we use this as t_obs

    Patients with metastasis or relapse status are flagged via the
    classifications file.

    Args:
        data_root: directory containing patient001.txt etc.
        classifications_path: optional path to classificationsofpatients.txt.
            If None, defaults to data_root/../classificationsofpatients.txt.
        min_obs: skip patients with fewer than this many usable PSA points.
        progression_psa_factor: define progression as PSA crossing this
            multiple of the patient's baseline PSA (Brady-Nicholls 2020 used
            1.20× for biochemical progression).

    Returns:
        BruchovskyCohort with one BruchovskyPatient per usable file.

    Notes:
        - Some PSA / testosterone fields are blank in the source files;
          those rows are dropped.
        - Schedule: we use the treatment column (1/0) directly. Real data
          has CPA+LEU (not abiraterone), but the binary on/off structure
          maps cleanly to our 3-pop K-shift model's Lambda(t) ∈ {0, 1}.
        - Progression detection here uses the patient's first PSA value as
          baseline; a more nuanced "pre-treatment baseline" would require
          additional metadata.
    """
    data_root = Path(data_root)
    if not data_root.is_absolute():
        # relative path resolved against the repo root, not cwd
        # (this module may be imported from many cwds)
        data_root = (Path(__file__).resolve().parents[2] / data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    if classifications_path is None:
        classifications_path = data_root.parent / "classificationsofpatients.txt"
    classifications_path = Path(classifications_path)

    # Parse classifications.
    relapse_ids: set[int] = set()
    metastasis_ids: set[int] = set()
    if classifications_path.exists():
        with classifications_path.open() as f:
            text = f.read()
        # Find the "With relapse:" block (multi-line)
        lines = text.split("\n")
        cur_label = None
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("With relapse:"):
                cur_label = "relapse"
                continue
            elif line_stripped.startswith("With metastasis"):
                cur_label = "metastasis"
                continue
            elif (line_stripped.startswith("Without")
                  or line_stripped.startswith("Other")
                  or line_stripped.endswith(":")):
                cur_label = None
                continue
            if cur_label is None or not line_stripped:
                continue
            # Parse comma-separated 3-digit IDs
            parts = [p.strip() for p in line_stripped.split(",")]
            for p in parts:
                try:
                    pid_int = int(p)
                    if cur_label == "relapse":
                        relapse_ids.add(pid_int)
                    elif cur_label == "metastasis":
                        metastasis_ids.add(pid_int)
                except ValueError:
                    continue

    patients: list[BruchovskyPatient] = []
    for fpath in sorted(data_root.glob("patient*.txt")):
        try:
            rows = []
            with fpath.open() as f:
                for line in f:
                    parts = [p.strip().strip('"') for p in line.strip().split(",")]
                    if len(parts) < 10:
                        continue
                    try:
                        psa_str = parts[4]
                        if not psa_str:
                            continue
                        psa = float(psa_str)
                        if psa < 0:
                            continue
                        treatment = int(float(parts[7])) if parts[7] else 0
                        # Use column 10 (alt day number) as the time axis
                        t_str = parts[9]
                        if not t_str:
                            continue
                        t_day = float(t_str)
                        rows.append((t_day, psa, treatment))
                    except (ValueError, IndexError):
                        continue
            if len(rows) < min_obs:
                continue
            rows.sort()  # ensure monotone time
            # De-duplicate same-day entries (keep first)
            seen_t = set()
            uniq = []
            for t, psa, u in rows:
                if t in seen_t:
                    continue
                seen_t.add(t)
                uniq.append((t, psa, u))
            if len(uniq) < min_obs:
                continue

            t_obs = np.array([r[0] for r in uniq])
            psa_obs = np.array([r[1] for r in uniq])
            u_schedule = np.array([float(r[2]) for r in uniq])

            # Re-base time so t_obs[0] = 0 (matches schema convention).
            t_obs = t_obs - t_obs[0]

            # Patient ID extracted from filename
            pid_str = fpath.stem.replace("patient", "")
            pid_int = int(pid_str)

            # Baseline PSA: first observation
            baseline = float(psa_obs[0])
            progression_thresh = progression_psa_factor * baseline
            prog_idx = np.where(psa_obs >= progression_thresh)[0]
            # Skip the first index since baseline by definition >= itself * 1.0
            prog_after_baseline = prog_idx[prog_idx > 0]
            if prog_after_baseline.size > 0:
                ttp_observed = float(t_obs[prog_after_baseline[0]])
                progressed = True
            else:
                ttp_observed = None
                progressed = False

            # Use clinical labels as a stronger ground-truth flag
            clinical_relapse = pid_int in relapse_ids
            clinical_metastasis = pid_int in metastasis_ids
            notes_parts = [f"source: dataTanaka/Bruchovsky_et_al/{fpath.name}"]
            if clinical_relapse:
                notes_parts.append("clinical_status: relapse")
                progressed = True  # override with clinical truth
            if clinical_metastasis:
                notes_parts.append("clinical_status: metastasis")

            patients.append(BruchovskyPatient(
                patient_id=f"bruchovsky_p{pid_int:03d}",
                t_obs=t_obs,
                psa_obs=psa_obs,
                u_schedule=u_schedule,
                baseline_psa=baseline,
                progression_observed=progressed,
                ttp_observed=ttp_observed,
                notes="; ".join(notes_parts),
            ))
        except Exception:  # noqa: BLE001 — per-patient parse failure tolerable
            continue

    import datetime as dt
    return BruchovskyCohort(
        patients=patients,
        source="dataTanaka/Bruchovsky_et_al (real)",
        date_acquired=dt.date.today().isoformat(),
    )


def load_shaw_et_al(
    data_root: Path | str = "data/raw/dataTanaka/Shaw_et_al",
    min_obs: int = 10,
    progression_psa_factor: float = 1.20,
) -> BruchovskyCohort:
    """Load the Shaw et al. (2007) cohort — second IADT trial in the dataTanaka archive.

    Same file schema as Bruchovsky_et_al except cols 3-4 (CPA/LEU dose) are
    typically blank — Shaw et al. used different drugs but the on/off treatment
    column is what we use anyway. 18 patients in the archive.

    Used for cross-cohort validation of the WP4 PE-vs-PA disagreement
    finding (experiment 20).
    """
    data_root = Path(data_root)
    if not data_root.is_absolute():
        data_root = (Path(__file__).resolve().parents[2] / data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    patients: list[BruchovskyPatient] = []
    for fpath in sorted(data_root.glob("patient*.txt")):
        try:
            rows = []
            with fpath.open() as f:
                for line in f:
                    parts = [p.strip().strip('"') for p in line.strip().split(",")]
                    if len(parts) < 9:
                        continue
                    try:
                        psa_str = parts[4]
                        if not psa_str:
                            continue
                        psa = float(psa_str)
                        if psa < 0:
                            continue
                        treatment = int(float(parts[7])) if parts[7] else 0
                        # Shaw files have only 9 columns (no alt day origin); use col 9.
                        t_str = parts[8]
                        if not t_str:
                            continue
                        t_day = float(t_str)
                        rows.append((t_day, psa, treatment))
                    except (ValueError, IndexError):
                        continue
            if len(rows) < min_obs:
                continue
            rows.sort()
            seen_t = set()
            uniq = []
            for t, psa, u in rows:
                if t in seen_t:
                    continue
                seen_t.add(t)
                uniq.append((t, psa, u))
            if len(uniq) < min_obs:
                continue

            t_obs = np.array([r[0] for r in uniq]) - uniq[0][0]
            psa_obs = np.array([r[1] for r in uniq])
            u_schedule = np.array([float(r[2]) for r in uniq])

            pid_str = fpath.stem.replace("patient", "")
            pid_int = int(pid_str)
            baseline = float(psa_obs[0])
            progression_thresh = progression_psa_factor * baseline
            prog_idx = np.where(psa_obs >= progression_thresh)[0]
            prog_after = prog_idx[prog_idx > 0]
            if prog_after.size > 0:
                ttp_observed = float(t_obs[prog_after[0]])
                progressed = True
            else:
                ttp_observed = None
                progressed = False

            patients.append(BruchovskyPatient(
                patient_id=f"shaw_p{pid_int:04d}",
                t_obs=t_obs,
                psa_obs=psa_obs,
                u_schedule=u_schedule,
                baseline_psa=baseline,
                progression_observed=progressed,
                ttp_observed=ttp_observed,
                notes=f"source: dataTanaka/Shaw_et_al/{fpath.name}",
            ))
        except Exception:  # noqa: BLE001
            continue

    import datetime as dt
    return BruchovskyCohort(
        patients=patients,
        source="dataTanaka/Shaw_et_al (real)",
        date_acquired=dt.date.today().isoformat(),
    )


def load_cohort_csv(path: Path | str, source: str = "real_data") -> BruchovskyCohort:
    """Load a real Bruchovsky cohort from CSV.

    Expected schema (one row per measurement):
    ``patient_id, t_obs_days, psa, u_schedule (0/1), baseline_psa,
       progression_observed (0/1), ttp_observed (or empty)``

    Patients are grouped by ``patient_id``; each group's measurements are
    sorted by ``t_obs_days``. The first row's ``baseline_psa`` /
    ``progression_observed`` / ``ttp_observed`` are taken as the patient-level
    metadata. Robust to small CSV-format variations.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"cohort CSV not found: {path}")

    rows_by_patient: dict[str, list[dict]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patient_id"]
            rows_by_patient.setdefault(pid, []).append(row)

    patients: list[BruchovskyPatient] = []
    for pid, rows in rows_by_patient.items():
        rows.sort(key=lambda r: float(r["t_obs_days"]))
        t_obs = np.array([float(r["t_obs_days"]) for r in rows])
        psa_obs = np.array([float(r["psa"]) for r in rows])
        u_schedule = np.array([float(r["u_schedule"]) for r in rows])
        first = rows[0]
        baseline = float(first["baseline_psa"]) if first.get("baseline_psa") else None
        progressed = bool(int(first.get("progression_observed", "0")))
        ttp_str = first.get("ttp_observed", "")
        ttp = float(ttp_str) if ttp_str.strip() else None
        patients.append(BruchovskyPatient(
            patient_id=pid, t_obs=t_obs, psa_obs=psa_obs, u_schedule=u_schedule,
            baseline_psa=baseline, progression_observed=progressed, ttp_observed=ttp,
        ))

    import datetime as dt
    return BruchovskyCohort(
        patients=patients,
        source=source,
        date_acquired=dt.date.today().isoformat(),
    )
