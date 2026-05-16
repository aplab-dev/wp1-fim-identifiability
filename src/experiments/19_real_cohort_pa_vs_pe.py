"""REAL Bruchovsky cohort: posterior-aware vs point-estimate decision comparison.

This is the M8-on-real-data deliverable for WP4.

Source data: http://www.nicholasbruchovsky.com/dataTanaka.zip — the same 72-
patient cohort that Brady-Nicholls 2020, Strobl 2022, and Gallagher 2025
fit. Loaded via realdata.load_dataTanaka().

For each real patient:
1. Fit theta via adaptive MH MCMC on the patient's PSA trajectory + treatment schedule.
2. Treat the resulting posterior samples as "the patient's posterior."
3. Compute three decisions:
   - Oracle: not available (we don't know real-patient theta_true).
   - Point-estimate: argmax over policies of TTP at the posterior mean theta.
   - Posterior-aware: argmax of E_theta[TTP] over the posterior.
4. Per-patient: which policy each method recommends + their disagreement.

Without an oracle for real patients, we cannot compute "accuracy". Instead we
report:
- Disagreement rate between PE and PA across the cohort.
- Distribution of expected TTP under PE-recommended vs PA-recommended policies.
- Per-patient: which patients have posterior-sensitive recommendations.

The disagreement rate is the empirical handle on "how often does posterior-
aware control differ from point-estimate control on real clinical fits?"

Output:
- ``results/figures/fig19_real_cohort_pa_vs_pe_{git_sha}_{date}.{png,pdf}``
- ``results/real_cohort_pa_vs_pe_summary_{git_sha}_{date}.json``
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from policies.at50 import AT50Policy  # noqa: E402
from policies.mtd import MTDPolicy  # noqa: E402
from realdata import (  # noqa: E402
    fit_patient_mcmc,
    load_dataTanaka,
)
from simulators.lv_3pop_kshift import LV3PopKShift, LV3PopParams  # noqa: E402
from zhang2017 import (  # noqa: E402
    ZhangPatientParams,
    run_zhang_patient,
    zhang_canonical_lv_params,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


PARAM_NAMES = ["r_T+", "r_TP", "r_T-", "α(T-,T+)", "α(T-,TP)", "K_TP_drop"]
_canon = zhang_canonical_lv_params()


def _build_lv_params(theta: np.ndarray) -> LV3PopParams:
    alpha = _canon.alpha.copy()
    alpha[2, 0] = max(theta[3], 0.01)
    alpha[2, 1] = max(theta[4], 0.01)
    return LV3PopParams(
        r_Tplus=max(theta[0], 1e-6), r_TP=max(theta[1], 1e-6), r_Tminus=max(theta[2], 1e-6),
        K_Tminus=_canon.K_Tminus, K_TP_max=_canon.K_TP_max,
        K_TP_drop=max(min(theta[5], _canon.K_TP_max - 1), 1.0),
        mu_max=_canon.mu_max, mu_drop=_canon.mu_drop,
        alpha=alpha,
    )


def expected_ttp_for_policy(theta: np.ndarray, policy_factory,
                            n_patients: int = 3,
                            rng: np.random.Generator | None = None) -> float | None:
    """Mean TTP across `n_patients` independent simulated patients at this theta."""
    if rng is None:
        rng = np.random.default_rng(0)
    try:
        params = ZhangPatientParams(lv_params=_build_lv_params(theta))
    except Exception:  # noqa: BLE001
        return None
    ttps = []
    ss = np.random.SeedSequence(int(rng.integers(0, 1_000_000)))
    for child in ss.spawn(n_patients):
        prng = np.random.Generator(np.random.PCG64(child))
        try:
            r = run_zhang_patient(params, policy_factory(), rng=prng)
            ttps.append(r["ttp"])
        except Exception:  # noqa: BLE001
            continue
    return float(np.mean(ttps)) if ttps else None


def evaluate_one_patient(patient, n_chains: int = 2, n_steps: int = 800,
                        burn_in: int = 300, n_post_samples: int = 30,
                        n_sim_per: int = 2, seed: int = 0) -> dict | None:
    """Real-patient PA-vs-PE decision comparison.

    1. Fit MCMC on this patient's PSA trajectory.
    2. Take a subsample of the posterior; compute expected TTP under MTD and AT50.
    3. PE: pick policy with higher expected TTP under the posterior mean.
    4. PA: pick policy with higher posterior-mean expected TTP.
    """
    rng = np.random.default_rng(seed)

    # 1) Fit MCMC
    try:
        result = fit_patient_mcmc(
            patient, n_chains=n_chains, n_steps=n_steps, burn_in=burn_in,
            thin=4, seed=seed,
        )
    except Exception:  # noqa: BLE001
        return None

    flat_samples = result.flat_samples()
    if flat_samples.shape[0] < n_post_samples:
        return None
    # Random subsample for policy evaluation (keeps cost bounded).
    sub_idx = rng.choice(flat_samples.shape[0], n_post_samples, replace=False)
    samples = flat_samples[sub_idx]
    posterior_mean = flat_samples.mean(axis=0)

    # 2) Point-estimate: TTP under each policy at posterior mean.
    ttp_mtd_pe = expected_ttp_for_policy(posterior_mean, MTDPolicy,
                                         n_patients=n_sim_per, rng=rng)
    ttp_at_pe = expected_ttp_for_policy(posterior_mean, AT50Policy,
                                        n_patients=n_sim_per, rng=rng)
    if ttp_mtd_pe is None or ttp_at_pe is None:
        return None
    pe_choice = "AT50" if ttp_at_pe > ttp_mtd_pe else "MTD"

    # 3) Posterior-aware: average TTP across posterior subsample for each policy.
    ttp_mtd_per_sample = []
    ttp_at_per_sample = []
    for theta in samples:
        m = expected_ttp_for_policy(theta, MTDPolicy, n_patients=n_sim_per, rng=rng)
        a = expected_ttp_for_policy(theta, AT50Policy, n_patients=n_sim_per, rng=rng)
        if m is not None and a is not None:
            ttp_mtd_per_sample.append(m)
            ttp_at_per_sample.append(a)
    if not ttp_mtd_per_sample:
        return None
    pa_expected_ttp_mtd = float(np.mean(ttp_mtd_per_sample))
    pa_expected_ttp_at = float(np.mean(ttp_at_per_sample))
    pa_choice = "AT50" if pa_expected_ttp_at > pa_expected_ttp_mtd else "MTD"

    # Probability AT50 wins TTP across the posterior
    pa_p_at50_wins = float(np.mean(
        np.array(ttp_at_per_sample) > np.array(ttp_mtd_per_sample)
    ))

    return {
        "patient_id": patient.patient_id,
        "n_obs": int(patient.n_obs()),
        "baseline_psa": float(patient.baseline),
        "clinical_progressed": bool(patient.progression_observed),
        "clinical_ttp": float(patient.ttp_observed) if patient.ttp_observed else None,
        "mcmc_rhat_max": float(result.rhat.max()),
        "mcmc_converged_at_1_50": bool(result.converged(rhat_threshold=1.50)),
        "posterior_mean_theta": posterior_mean.tolist(),
        "pe_choice": pe_choice,
        "pe_ttp_mtd_d": ttp_mtd_pe,
        "pe_ttp_at50_d": ttp_at_pe,
        "pa_choice": pa_choice,
        "pa_expected_ttp_mtd_d": pa_expected_ttp_mtd,
        "pa_expected_ttp_at50_d": pa_expected_ttp_at,
        "pa_p_at50_wins": pa_p_at50_wins,
        "pe_pa_disagree": pe_choice != pa_choice,
    }


def main(n_patients: int | None = None, n_chains: int = 2, n_steps: int = 800,
         burn_in: int = 300, n_post_samples: int = 25, n_sim_per: int = 2,
         seed: int = 0) -> None:
    warnings.filterwarnings("ignore")
    log.info("REAL-cohort PA-vs-PE decision comparison")
    log.info("Source: dataTanaka/Bruchovsky_et_al (Nicholas Bruchovsky's public repository)")

    cohort = load_dataTanaka()
    log.info(f"Loaded {cohort.n_patients} real patients; clinical progression rate {cohort.progression_rate():.0%}")

    if n_patients is not None and n_patients < cohort.n_patients:
        # Subsample to make runtime manageable. Keep first N to ensure determinism.
        log.info(f"Subsampling to first {n_patients} patients for runtime")
        cohort.patients[:] = cohort.patients[:n_patients]

    log.info(f"  per-patient settings: n_chains={n_chains}, n_steps={n_steps}, "
             f"burn_in={burn_in}, n_post_samples={n_post_samples}, n_sim_per={n_sim_per}")

    results = []
    for i, patient in enumerate(cohort.patients):
        log.info(f"  [{i+1}/{len(cohort.patients)}] {patient.patient_id} "
                 f"(n_obs={patient.n_obs()})")
        try:
            res = evaluate_one_patient(
                patient, n_chains=n_chains, n_steps=n_steps, burn_in=burn_in,
                n_post_samples=n_post_samples, n_sim_per=n_sim_per,
                seed=seed * 1000 + i,
            )
            if res is None:
                log.warning(f"     -> evaluation failed")
                continue
            results.append(res)
            log.info(
                f"     PE={res['pe_choice']}, PA={res['pa_choice']}, "
                f"P(AT50>MTD)={res['pa_p_at50_wins']:.0%}, "
                f"rhat_max={res['mcmc_rhat_max']:.2f}"
            )
        except Exception as e:  # noqa: BLE001
            log.warning(f"     exception: {e}")
            continue

    if not results:
        log.error("No patients evaluated successfully.")
        return

    # Aggregate
    n_total = len(results)
    n_pe_at = sum(1 for r in results if r["pe_choice"] == "AT50")
    n_pa_at = sum(1 for r in results if r["pa_choice"] == "AT50")
    n_disagree = sum(1 for r in results if r["pe_pa_disagree"])
    n_converged = sum(1 for r in results if r["mcmc_converged_at_1_50"])

    log.info(f"REAL-cohort summary across {n_total} patients:")
    log.info(f"  MCMC convergence rate (rhat<1.50): {n_converged}/{n_total} = {n_converged/n_total:.0%}")
    log.info(f"  PE recommends AT50:                {n_pe_at}/{n_total} = {n_pe_at/n_total:.0%}")
    log.info(f"  PA recommends AT50:                {n_pa_at}/{n_total} = {n_pa_at/n_total:.0%}")
    log.info(f"  PE-vs-PA disagreement:             {n_disagree}/{n_total} = {n_disagree/n_total:.0%}")

    p_at_pa = np.array([r["pa_p_at50_wins"] for r in results])
    n_posterior_sensitive = int(np.sum((p_at_pa > 0.10) & (p_at_pa < 0.90)))
    log.info(f"  Posterior-sensitive patients (10% < P < 90%): {n_posterior_sensitive}/{n_total}")

    # --- Figure ---
    fig = plt.figure(figsize=(15, 10))
    fig.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.32)

    # Panel 1: histogram of pa_p_at50_wins (per-patient posterior probability AT50 better)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(p_at_pa, bins=20, color="tab:red", alpha=0.7, edgecolor="black")
    ax1.axvline(0.5, color="black", linestyle="--", linewidth=1.0, label="coin-flip")
    ax1.set_xlabel("P(AT50 beats MTD on TTP) per patient")
    ax1.set_ylabel("Patient count")
    ax1.set_title(f"[A] Posterior-aware AT50-preference\n(N={n_total} real patients)", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: PE vs PA TTP advantage scatter
    ax2 = fig.add_subplot(gs[0, 1])
    pe_adv = np.array([r["pe_ttp_at50_d"] - r["pe_ttp_mtd_d"] for r in results]) / 30
    pa_adv = np.array([r["pa_expected_ttp_at50_d"] - r["pa_expected_ttp_mtd_d"] for r in results]) / 30
    disagree_mask = np.array([r["pe_pa_disagree"] for r in results])
    ax2.scatter(pe_adv[~disagree_mask], pa_adv[~disagree_mask],
                c="tab:gray", s=40, alpha=0.6, label="PE = PA", edgecolor="none")
    ax2.scatter(pe_adv[disagree_mask], pa_adv[disagree_mask],
                c="tab:orange", s=60, alpha=0.8, label=f"PE ≠ PA (n={n_disagree})", edgecolor="black")
    diag = np.linspace(min(pe_adv.min(), pa_adv.min()), max(pe_adv.max(), pa_adv.max()), 100)
    ax2.plot(diag, diag, "k--", alpha=0.4, linewidth=0.7, label="PE = PA")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axvline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("PE AT50 advantage (months)")
    ax2.set_ylabel("PA AT50 advantage (months)")
    ax2.set_title(f"[B] PE vs PA per patient", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: convergence diagnostic distribution
    ax3 = fig.add_subplot(gs[0, 2])
    rhats = [r["mcmc_rhat_max"] for r in results]
    ax3.hist(np.minimum(rhats, 10), bins=30, color="tab:blue", alpha=0.7, edgecolor="black")
    ax3.axvline(1.10, color="tab:red", linestyle="--", linewidth=1.0, label="rhat=1.10")
    ax3.axvline(1.50, color="tab:orange", linestyle="--", linewidth=1.0, label="rhat=1.50")
    ax3.set_xlabel("MCMC rhat_max (clipped at 10)")
    ax3.set_ylabel("Patient count")
    ax3.set_title(f"[C] Convergence (real-data MH)", fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: PE vs PA recommendation breakdown by clinical progression status
    ax4 = fig.add_subplot(gs[1, 0])
    cats = ["Clinically\nstable", "Clinically\nprogressed"]
    progressed = np.array([r["clinical_progressed"] for r in results])
    breakdowns = []
    for cohort_mask, label in [(~progressed, cats[0]), (progressed, cats[1])]:
        if cohort_mask.sum() == 0:
            breakdowns.append([0, 0, 0, 0])
            continue
        n_in_cohort = int(cohort_mask.sum())
        pe_at = sum(1 for r, m in zip(results, cohort_mask) if m and r["pe_choice"] == "AT50")
        pa_at = sum(1 for r, m in zip(results, cohort_mask) if m and r["pa_choice"] == "AT50")
        breakdowns.append([n_in_cohort - pe_at, pe_at, n_in_cohort - pa_at, pa_at])
    width = 0.4
    x = np.arange(len(cats))
    pe_at_counts = [b[1] for b in breakdowns]
    pe_mtd_counts = [b[0] for b in breakdowns]
    pa_at_counts = [b[3] for b in breakdowns]
    pa_mtd_counts = [b[2] for b in breakdowns]
    ax4.bar(x - width / 2, pe_at_counts, width, color="tab:red", alpha=0.6, label="PE: AT50")
    ax4.bar(x - width / 2, pe_mtd_counts, width, bottom=pe_at_counts, color="tab:blue", alpha=0.6, label="PE: MTD")
    ax4.bar(x + width / 2, pa_at_counts, width, color="tab:red", alpha=0.9, label="PA: AT50", hatch="//")
    ax4.bar(x + width / 2, pa_mtd_counts, width, bottom=pa_at_counts, color="tab:blue", alpha=0.9, label="PA: MTD", hatch="//")
    ax4.set_xticks(x)
    ax4.set_xticklabels(cats)
    ax4.set_ylabel("Patient count")
    ax4.set_title("[D] Recommendation by clinical status", fontsize=10)
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3, axis="y")

    # Panel 5: posterior probability vs MCMC convergence (does poor convergence predict sensitivity?)
    ax5 = fig.add_subplot(gs[1, 1])
    rhats_arr = np.array(rhats)
    color_disagree = ["tab:orange" if r["pe_pa_disagree"] else "tab:gray" for r in results]
    ax5.scatter(np.minimum(rhats_arr, 10), p_at_pa, c=color_disagree, s=40, alpha=0.7, edgecolor="none")
    ax5.axhline(0.5, color="black", linewidth=0.5, alpha=0.5)
    ax5.axvline(1.50, color="tab:orange", linestyle="--", linewidth=0.8, alpha=0.6)
    ax5.set_xlabel("rhat_max (clipped at 10)")
    ax5.set_ylabel("P(AT50 beats MTD)")
    ax5.set_title("[E] Convergence vs posterior preference", fontsize=10)
    ax5.grid(True, alpha=0.3)

    # Panel 6: summary text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    summary_text = (
        f"REAL BRUCHOVSKY COHORT (n={n_total})\n\n"
        f"Source: nicholasbruchovsky.com/dataTanaka.zip\n"
        f"Same cohort fit by Brady-Nicholls 2020,\n"
        f"Strobl 2022, and Gallagher 2025.\n\n"
        f"MCMC convergence (rhat<1.50):\n"
        f"  {n_converged}/{n_total} = {n_converged/n_total:.0%}\n\n"
        f"Recommendations:\n"
        f"  PE  recommends AT50: {n_pe_at}/{n_total} = {n_pe_at/n_total:.0%}\n"
        f"  PA  recommends AT50: {n_pa_at}/{n_total} = {n_pa_at/n_total:.0%}\n\n"
        f"PE-vs-PA disagreement: {n_disagree}/{n_total} = {n_disagree/n_total:.0%}\n"
        f"Posterior-sensitive: {n_posterior_sensitive}/{n_total}\n"
        f"  (10% < P(AT50 wins) < 90%)\n\n"
        f"This is the WP4 main empirical figure.\n"
        f"Phase 3 §3.3 + §3.4 deliverable on REAL data."
    )
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    fig.suptitle(
        "REAL Bruchovsky cohort — posterior-aware vs point-estimate clinical decisions",
        fontsize=12,
    )

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True,
        ).strip()
    except subprocess.CalledProcessError:
        sha = "uncommitted"
    date = dt.date.today().isoformat()
    base = _REPO_ROOT / "results" / "figures" / f"fig19_real_cohort_pa_vs_pe_{sha}_{date}"
    fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {base}.png and .pdf")

    summary = {
        "experiment": "real_cohort_pa_vs_pe",
        "git_sha": sha,
        "date": date,
        "data_source": "http://www.nicholasbruchovsky.com/dataTanaka.zip",
        "n_patients_loaded": int(cohort.n_patients),
        "n_patients_evaluated": int(n_total),
        "settings": {
            "n_chains": int(n_chains),
            "mh_n_steps": int(n_steps),
            "burn_in": int(burn_in),
            "n_post_samples": int(n_post_samples),
            "n_sim_per": int(n_sim_per),
        },
        "headline": {
            "n_pe_recommends_at50": int(n_pe_at),
            "n_pa_recommends_at50": int(n_pa_at),
            "n_pe_pa_disagree": int(n_disagree),
            "pe_pa_disagreement_rate": n_disagree / n_total,
            "n_posterior_sensitive": int(n_posterior_sensitive),
            "n_mcmc_converged_rhat_under_1_50": int(n_converged),
        },
        "per_patient_results": results,
    }
    summary_path = _REPO_ROOT / "results" / f"real_cohort_pa_vs_pe_summary_{sha}_{date}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REAL Bruchovsky cohort PA-vs-PE")
    parser.add_argument("--n-patients", type=int, default=None,
                        help="Subsample to first N patients; default = all 72")
    parser.add_argument("--n-chains", type=int, default=2)
    parser.add_argument("--mh-n-steps", type=int, default=800, dest="n_steps")
    parser.add_argument("--burn-in", type=int, default=300)
    parser.add_argument("--n-post-samples", type=int, default=25)
    parser.add_argument("--n-sim-per", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(
        n_patients=args.n_patients, n_chains=args.n_chains,
        n_steps=args.n_steps, burn_in=args.burn_in,
        n_post_samples=args.n_post_samples, n_sim_per=args.n_sim_per,
        seed=args.seed,
    )
