"""Cross-tribe comparison: AT50 ≫ MTD on both 2-pop multdeath AND 3-pop K-shift?

Stage 2.6 sub-thread B. The Phase 1 literature review found that the field
splits into two modeling tribes:

- **Clinical tribe** (Zhang 2017, Cunningham 2020, West 2020) uses a 3-pop
  K-shift model. Stage 2.4 reproduced AT50 ≫ MTD on this model.
- **Theory tribe** (Strobl 2021, Gallagher 2025, Wang & Lei 2025) uses a
  2-pop multiplicative-death model. Stage 2.1 simulator + Stage 2.3
  policies are ready; we just need a cohort runner.

This experiment runs the same three policies (No-tx / MTD / AT50) on both
models with cohort variation, and compares the headline finding side-by-
side. If both tribes agree (AT50 > MTD on TTP), the result is robust to
model choice. If they disagree, that's a research-question-worthy finding.

The 2-pop multdeath patient setup parallels Zhang's:
- Canonical params: regime A weak competition (alpha=0.7, beta=0.6),
  r_S=0.05, r_R=0.04, K=1.0, d=1.5.
- IC at 25% of untreated equilibrium (S, R) ratio with small perturbation.
- Baseline PSA = rho * (S0 + R0) / phi (Zhang-style PSA filter).
- Progression: PSA crosses 1.20 * baseline.
- 28-day decision interval.
- Simulation horizon 1500 days.

Output:
- ``results/figures/fig06_cross_tribe_comparison_{git_sha}_{date}.{png,pdf}``
- ``results/cross_tribe_summary_{git_sha}_{date}.json``
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from policies.at50 import AT50Policy  # noqa: E402
from policies.base import Observation  # noqa: E402
from policies.cohort_runner import CohortRunner  # noqa: E402
from policies.mtd import MTDPolicy  # noqa: E402
from policies.no_treatment import NoTreatmentPolicy  # noqa: E402
from simulators.lv_2pop_multdeath import LV2PopMultDeath, LV2PopParams  # noqa: E402
from simulators.psa_dynamics import PSAParams, psa_steady_state  # noqa: E402
from zhang2017 import run_zhang_patient, zhang_2017_sampler  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------- 2-pop multdeath patient runner (theory-tribe analog) ----------


def theory2pop_sampler(rng: np.random.Generator) -> dict:  # noqa: ARG001
    """Returns canonical theory-tribe params (one set per cohort).

    Cohort variation enters via ic_perturbation_std in the runner using
    the per-patient rng. Same pattern as zhang_2017_sampler.
    """
    return {
        "lv_params": LV2PopParams(r_S=0.05, r_R=0.04, alpha=0.7, beta=0.6, K=1.0, d=1.5),
        "psa_params": PSAParams(),
        "x0_canonical": (0.6, 0.006),  # S/R ratio = 100:1 (Zhang-like)
        "ic_perturbation_std": 0.10,
        "progression_psa_threshold": 1.20,
        "t_max": 1500.0,
        "decision_interval": 28.0,
    }


def run_2pop_patient(
    params: dict,
    policy,
    rng: np.random.Generator | None = None,
) -> dict:
    """Single-patient runner for the 2-pop multdeath model.

    Mirrors zhang2017.run_zhang_patient signature.
    """
    sim = LV2PopMultDeath(params["lv_params"])
    psa_params = params["psa_params"]
    t_max = params["t_max"]
    decision_interval = params["decision_interval"]

    # IC with perturbation
    x0 = np.array(params["x0_canonical"], dtype=float)
    pert_std = params["ic_perturbation_std"]
    if pert_std > 0 and rng is not None:
        x0 = x0 * np.exp(rng.normal(0, pert_std, size=2))
    if np.any(x0 <= 0):
        raise ValueError(f"IC has non-positive component: {x0}")

    baseline_psa = psa_steady_state(float(np.sum(x0)), psa_params)
    progression_psa = params["progression_psa_threshold"] * baseline_psa

    current = x0.copy()
    current_psa = baseline_psa
    t_now = 0.0
    cum_dose = 0.0
    ttp = t_max
    progressed = False
    t_chunks = [np.array([0.0])]
    psa_chunks = [np.array([baseline_psa])]

    while t_now < t_max:
        obs = Observation(t=t_now, psa=current_psa, baseline_psa=baseline_psa)
        u = float(policy(obs))
        t_end = min(t_now + decision_interval, t_max)

        def rhs(t: float, y: np.ndarray, u_const: float = u) -> np.ndarray:
            S, R, PSA = y
            dS, dR = sim.dynamics(t, np.array([S, R]), u_const)
            dPSA = psa_params.rho * (S + R) - psa_params.phi * PSA
            return np.array([dS, dR, dPSA])

        # Try LSODA, fall back to BDF
        sol = None
        for method in ("LSODA", "BDF"):
            try:
                trial = solve_ivp(
                    rhs,
                    t_span=(t_now, t_end),
                    y0=np.array([current[0], current[1], current_psa]),
                    t_eval=np.linspace(t_now, t_end, 30),
                    method=method,
                    rtol=1e-6, atol=1e-3,
                )
                if trial.success:
                    sol = trial
                    break
            except Exception:  # noqa: BLE001
                continue
        if sol is None:
            raise RuntimeError("2-pop solve_ivp failed under LSODA + BDF")

        # Progression check
        chunk_psa = sol.y[2]
        crossings = np.where(chunk_psa >= progression_psa)[0]
        if crossings.size > 0:
            ttp_chunk = float(sol.t[crossings[0]])
            cum_dose += u * (ttp_chunk - t_now)
            ttp = ttp_chunk
            progressed = True
            t_chunks.append(sol.t[1:])
            psa_chunks.append(chunk_psa[1:])
            t_now = t_end
            break

        cum_dose += u * (t_end - t_now)
        current = np.array([sol.y[0, -1], sol.y[1, -1]])
        current_psa = float(sol.y[2, -1])
        t_chunks.append(sol.t[1:])
        psa_chunks.append(chunk_psa[1:])
        t_now = t_end

    return {
        "ttp": ttp,
        "cumulative_dose": cum_dose,
        "progressed": progressed,
        "baseline_psa": baseline_psa,
    }


# ---------- main experiment ----------


def kaplan_meier_curve(ttps: np.ndarray, progressed: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    return np.array([
        np.mean(~((ttps <= t) & progressed)) for t in t_grid
    ])


def main(seed: int = 0, n_patients: int = 30, t_max: float = 1500.0) -> None:
    warnings.filterwarnings("ignore")

    log.info(f"Cross-tribe comparison: 2-pop multdeath vs 3-pop K-shift, N={n_patients}/arm")

    policies = {
        "No-treatment": NoTreatmentPolicy,
        "MTD": MTDPolicy,
        "AT50": AT50Policy,
    }
    arm_colors = {"No-treatment": "tab:gray", "MTD": "tab:blue", "AT50": "tab:red"}

    # Theory-tribe (2-pop) cohorts
    theory_results = {}
    log.info("Theory tribe (2-pop multdeath):")
    for name, policy_cls in policies.items():
        runner = CohortRunner(
            run_one_patient=run_2pop_patient,
            param_sampler=theory2pop_sampler,
            n_patients=n_patients,
            seed=seed,
        )
        theory_results[name] = runner.run(policy_factory=policy_cls)
        ttps = theory_results[name].ttp_array()
        doses = theory_results[name].cumulative_dose_array()
        log.info(
            f"  {name:14s}  TTP median={np.median(ttps):5.0f}d ({np.median(ttps)/30:.1f} mo) "
            f"prog={theory_results[name].progression_rate():.0%}  drug_frac={np.mean(doses)/np.mean(ttps):.1%}"
        )

    # Clinical-tribe (3-pop K-shift, Zhang) cohorts — re-run for fairness
    clinical_results = {}
    log.info("Clinical tribe (3-pop K-shift, Zhang 2017):")
    for name, policy_cls in policies.items():
        runner = CohortRunner(
            run_one_patient=run_zhang_patient,
            param_sampler=zhang_2017_sampler,
            n_patients=n_patients,
            seed=seed,
        )
        clinical_results[name] = runner.run(policy_factory=policy_cls)
        ttps = clinical_results[name].ttp_array()
        doses = clinical_results[name].cumulative_dose_array()
        log.info(
            f"  {name:14s}  TTP median={np.median(ttps):5.0f}d ({np.median(ttps)/30:.1f} mo) "
            f"prog={clinical_results[name].progression_rate():.0%}  drug_frac={np.mean(doses)/np.mean(ttps):.1%}"
        )

    # --- Summary ---
    summary = {"theory_tribe": {}, "clinical_tribe": {}}
    for name in policies:
        for label, results in [("theory_tribe", theory_results), ("clinical_tribe", clinical_results)]:
            ttps = results[name].ttp_array()
            doses = results[name].cumulative_dose_array()
            summary[label][name] = {
                "n": int(len(results[name].per_patient)),
                "ttp_median_d": float(np.median(ttps)),
                "ttp_iqr_d": [float(np.percentile(ttps, 25)), float(np.percentile(ttps, 75))],
                "progression_rate": float(results[name].progression_rate()),
                "drug_fraction_mean": float(np.mean(doses) / np.mean(ttps)) if np.mean(ttps) > 0 else 0.0,
            }

    # Cross-tribe headline check
    theory_at50_better = (
        summary["theory_tribe"]["AT50"]["ttp_median_d"]
        > summary["theory_tribe"]["MTD"]["ttp_median_d"]
    )
    clinical_at50_better = (
        summary["clinical_tribe"]["AT50"]["ttp_median_d"]
        > summary["clinical_tribe"]["MTD"]["ttp_median_d"]
    )
    summary["headline"] = {
        "theory_at50_beats_mtd": bool(theory_at50_better),
        "clinical_at50_beats_mtd": bool(clinical_at50_better),
        "cross_tribe_agreement": bool(theory_at50_better == clinical_at50_better),
    }

    # --- Figure ---
    fig = plt.figure(figsize=(15, 9))
    fig.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.32)

    t_grid = np.linspace(0, t_max, 200)
    for row, (label, results, title) in enumerate([
        ("theory", theory_results, "Theory tribe (2-pop multdeath)"),
        ("clinical", clinical_results, "Clinical tribe (3-pop K-shift, Zhang 2017)"),
    ]):
        # KM curves
        ax_km = fig.add_subplot(gs[row, 0])
        for name, cohort in results.items():
            ttps = cohort.ttp_array()
            progressed = np.array([r["progressed"] for r in cohort.per_patient])
            km = kaplan_meier_curve(ttps, progressed, t_grid)
            median_mo = np.median(ttps) / 30
            ax_km.plot(t_grid, km, color=arm_colors[name], linewidth=2.0,
                       label=f"{name} (median {median_mo:.0f} mo)")
        ax_km.set_xlabel("Time (days)")
        ax_km.set_ylabel("Fraction not progressed")
        ax_km.set_ylim(-0.02, 1.05)
        ax_km.set_xlim(0, t_max)
        ax_km.set_title(f"{title}", fontsize=11)
        ax_km.grid(True, alpha=0.3)
        ax_km.legend(loc="lower left", fontsize=10)

        # Dose vs TTP scatter
        ax_sc = fig.add_subplot(gs[row, 1])
        for name, cohort in results.items():
            ttps = cohort.ttp_array()
            doses = cohort.cumulative_dose_array()
            ax_sc.scatter(doses, ttps, c=arm_colors[name], s=40, alpha=0.6, edgecolor="none", label=name)
        ax_sc.set_xlabel("Cumulative drug exposure (drug-days)")
        ax_sc.set_ylabel("TTP (days)")
        ax_sc.set_title(f"Dose vs TTP — {title}", fontsize=11)
        ax_sc.grid(True, alpha=0.3)
        ax_sc.legend(loc="lower right", fontsize=9)

    headline_text = (
        f"Cross-tribe headline: AT50 > MTD on both → {summary['headline']['cross_tribe_agreement']}"
    )
    fig.suptitle(
        f"Cross-tribe comparison (Stage 2.6) — {headline_text}",
        fontsize=12,
    )

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True,
        ).strip()
    except subprocess.CalledProcessError:
        sha = "uncommitted"
    date = dt.date.today().isoformat()
    base = _REPO_ROOT / "results" / "figures" / f"fig06_cross_tribe_comparison_{sha}_{date}"
    fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {base}.png and .pdf")

    summary_path = (
        _REPO_ROOT / "results" / f"cross_tribe_summary_{sha}_{date}.json"
    )
    full_summary = {
        "experiment": "cross_tribe_comparison",
        "git_sha": sha,
        "date": date,
        "n_patients": n_patients,
        "t_max_days": t_max,
        "seed": seed,
        **summary,
    }
    with summary_path.open("w") as f:
        json.dump(full_summary, f, indent=2)
    log.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-tribe AT50 vs MTD comparison")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--n-patients", type=int, default=30, help="Cohort size per arm")
    parser.add_argument("--t-max", type=float, default=1500.0, help="Sim horizon (days)")
    args = parser.parse_args()
    main(seed=args.seed, n_patients=args.n_patients, t_max=args.t_max)
