"""Hierarchical Bayesian fit pooling across the 71 Bruchovsky patients.

Closes WP1 §7.2 limitation #1 ("single-patient FIM is a worst-case-per-patient
bound, not a cohort-fitting bound").

Pipeline:
1. Load Bruchovsky cohort.
2. For each patient, run adaptive MH MCMC (same as exp 19).
3. Convert per-patient posterior samples to log-space Gaussian summaries.
4. Run NUTS on the closed-form Gaussian hierarchical model
   (`realdata.hierarchical`) — no diffrax in the inner loop, so the
   warmup-hang issue from `per_patient_hmc.py` does not apply.
5. Quantify per-parameter shrinkage (pooled vs unpooled posterior std).

Outputs:
- ``results/figures/fig22_hierarchical_bruchovsky_{git_sha}_{date}.{png,pdf}``
- ``results/hierarchical_bruchovsky_summary_{git_sha}_{date}.json``

Headline metric: median per-parameter shrinkage factor (pooled / unpooled std).
A factor < 1 means pooling tightened the per-patient posterior.
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

from realdata import (  # noqa: E402
    PatientSummary,
    compare_pooled_vs_unpooled,
    fit_patient_mcmc,
    hierarchical_fit,
    load_dataTanaka,
    load_shaw_et_al,
    per_patient_summaries,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PARAM_NAMES = ["r_T+", "r_TP", "r_T-", "α(T-,T+)", "α(T-,TP)", "K_TP_drop"]


def collect_per_patient_posteriors(
    cohort, n_chains: int = 2, n_steps: int = 600,
    burn_in: int = 200, thin: int = 4, seed: int = 0,
) -> list[PatientSummary]:
    """Fit MH-MCMC per patient and return Gaussian-summary objects."""
    summaries = []
    n_failed = 0
    for i, patient in enumerate(cohort.patients):
        log.info(f"  [{i+1}/{len(cohort.patients)}] {patient.patient_id} "
                 f"(n_obs={patient.n_obs()})")
        try:
            result = fit_patient_mcmc(
                patient, n_chains=n_chains, n_steps=n_steps, burn_in=burn_in,
                thin=thin, seed=seed * 1000 + i,
            )
            summary_list = per_patient_summaries(
                [result.flat_samples()], patient_ids=[patient.patient_id],
            )
            if summary_list:
                summaries.append(summary_list[0])
                log.info(
                    f"     ok: log_θ_std (mean over params) = "
                    f"{summary_list[0].log_theta_std.mean():.3f}"
                )
        except Exception as e:  # noqa: BLE001
            log.warning(f"     fit failed: {e}")
            n_failed += 1
            continue
    log.info(f"Collected {len(summaries)} per-patient summaries; {n_failed} failed")
    return summaries


def main(n_patients: int | None = None, n_chains: int = 2, n_steps: int = 600,
         burn_in: int = 200, n_h_warmup: int = 500, n_h_samples: int = 1000,
         n_h_chains: int = 4, seed: int = 0,
         cohort_name: str = "bruchovsky") -> None:
    warnings.filterwarnings("ignore")
    log.info(f"Hierarchical Bayesian fit on {cohort_name} cohort")

    if cohort_name.lower() == "shaw":
        cohort = load_shaw_et_al()
    else:
        cohort = load_dataTanaka()
    log.info(f"Loaded {cohort.n_patients} real patients")

    if n_patients is not None and n_patients < cohort.n_patients:
        log.info(f"Subsampling to first {n_patients} patients")
        cohort.patients[:] = cohort.patients[:n_patients]

    log.info("Step 1/2: per-patient MH-MCMC fits")
    summaries = collect_per_patient_posteriors(
        cohort, n_chains=n_chains, n_steps=n_steps,
        burn_in=burn_in, seed=seed,
    )
    if not summaries:
        log.error("No patient summaries collected; aborting.")
        return

    log.info(f"Step 2/2: hierarchical NUTS over {len(summaries)} patients")
    h = hierarchical_fit(
        summaries, n_chains=n_h_chains, n_samples=n_h_samples,
        n_warmup=n_h_warmup, seed=seed, progress_bar=False,
    )
    log.info(f"  hierarchical R-hat μ: {h.rhat_pop_mean.tolist()}")
    log.info(f"  hierarchical R-hat σ: {h.rhat_pop_std.tolist()}")
    log.info(f"  converged (R-hat < 1.10): {h.converged()}")

    # Population-level posterior summary
    mu_mean = h.pop_mean_samples.mean(axis=0)
    mu_std = h.pop_mean_samples.std(axis=0)
    sigma_mean = h.pop_std_samples.mean(axis=0)
    sigma_std = h.pop_std_samples.std(axis=0)
    log.info("Population-level posterior (log θ space):")
    for k, name in enumerate(PARAM_NAMES):
        log.info(
            f"  {name:<12} μ = {mu_mean[k]:+.3f} ± {mu_std[k]:.3f}, "
            f"σ_pop = {sigma_mean[k]:.3f} ± {sigma_std[k]:.3f}"
        )

    # Shrinkage
    shrink = compare_pooled_vs_unpooled(summaries, h)
    log.info("Median per-parameter shrinkage (pooled / unpooled std):")
    for k, name in enumerate(PARAM_NAMES):
        log.info(f"  {name:<12} {shrink['median_shrinkage_per_param'][k]:.3f}")

    # --- Figure ---
    n_pat = len(summaries)
    fig = plt.figure(figsize=(15, 9))
    fig.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.32)

    # Panel A: per-parameter forest plot of unpooled per-patient log_θ_mean ± std
    ax_a = fig.add_subplot(gs[0, 0])
    log_theta_mean = np.array([s.log_theta_mean for s in summaries])
    log_theta_std = np.array([s.log_theta_std for s in summaries])
    for k, name in enumerate(PARAM_NAMES):
        x = log_theta_mean[:, k]
        ax_a.scatter(np.full(n_pat, k), x, s=8, c="tab:gray", alpha=0.4, edgecolor="none")
        ax_a.errorbar(k, x.mean(), yerr=x.std(), fmt="o",
                      color="tab:red", capsize=3, markersize=6, alpha=0.9)
    ax_a.set_xticks(range(6))
    ax_a.set_xticklabels(PARAM_NAMES, rotation=30, ha="right", fontsize=8)
    ax_a.set_ylabel("log θ (per-patient)")
    ax_a.set_title(f"[A] Per-patient posterior means\n({n_pat} Bruchovsky patients)", fontsize=10)
    ax_a.grid(True, alpha=0.3)
    ax_a.axhline(0, color="black", linewidth=0.5, alpha=0.5)

    # Panel B: population-level posterior on μ (with prior overlay)
    ax_b = fig.add_subplot(gs[0, 1])
    for k, name in enumerate(PARAM_NAMES):
        samples = h.pop_mean_samples[:, k]
        # KDE-ish histogram
        ax_b.hist(samples, bins=40, alpha=0.4, label=name, density=True)
    ax_b.set_xlabel("μ_k posterior (log θ)")
    ax_b.set_ylabel("Density")
    ax_b.set_title("[B] Population mean (μ) posterior", fontsize=10)
    ax_b.legend(fontsize=7, ncol=2)
    ax_b.grid(True, alpha=0.3)

    # Panel C: population-level posterior on σ_pop
    ax_c = fig.add_subplot(gs[0, 2])
    for k, name in enumerate(PARAM_NAMES):
        samples = h.pop_std_samples[:, k]
        ax_c.hist(samples, bins=40, alpha=0.4, label=name, density=True)
    ax_c.set_xlabel("σ_pop posterior (log θ space)")
    ax_c.set_ylabel("Density")
    ax_c.set_title("[C] Population SD (σ_pop) posterior", fontsize=10)
    ax_c.legend(fontsize=7, ncol=2)
    ax_c.grid(True, alpha=0.3)

    # Panel D: per-parameter shrinkage box
    ax_d = fig.add_subplot(gs[1, 0])
    shrink_arr = np.array(shrink["shrinkage_factor_per_patient_per_param"])
    bp = ax_d.boxplot([shrink_arr[:, k] for k in range(6)],
                      tick_labels=PARAM_NAMES, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("tab:purple")
        patch.set_alpha(0.6)
    ax_d.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="no shrinkage")
    ax_d.set_xticklabels(PARAM_NAMES, rotation=30, ha="right", fontsize=8)
    ax_d.set_ylabel("pooled std / unpooled std")
    ax_d.set_title("[D] Per-parameter shrinkage", fontsize=10)
    ax_d.legend(fontsize=8)
    ax_d.grid(True, alpha=0.3)

    # Panel E: scatter unpooled vs pooled std for one representative parameter
    ax_e = fig.add_subplot(gs[1, 1])
    pooled = np.array(shrink["pooled_std_per_patient_per_param"])
    sigma_obs = log_theta_std
    k_sel = 3  # α(T-,T+) — the unidentifiable direction
    ax_e.scatter(sigma_obs[:, k_sel], pooled[:, k_sel],
                 c="tab:purple", s=30, alpha=0.6, edgecolor="black")
    diag = np.linspace(0, sigma_obs[:, k_sel].max(), 100)
    ax_e.plot(diag, diag, "k--", alpha=0.4, linewidth=0.7, label="no shrinkage")
    ax_e.set_xlabel("Unpooled σ (per-patient MCMC)")
    ax_e.set_ylabel("Pooled σ (under hierarchical posterior)")
    ax_e.set_title(f"[E] {PARAM_NAMES[k_sel]} pooling", fontsize=10)
    ax_e.legend(fontsize=8)
    ax_e.grid(True, alpha=0.3)

    # Panel F: text summary
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")
    median_shrink = shrink["median_shrinkage_per_param"]
    summary_text = (
        f"HIERARCHICAL BAYES (n={n_pat})\n\n"
        f"Source: dataTanaka.zip (Bruchovsky)\n\n"
        f"R-hat (μ): max {h.rhat_pop_mean.max():.2f}\n"
        f"R-hat (σ): max {h.rhat_pop_std.max():.2f}\n"
        f"Converged: {h.converged()}\n\n"
        f"Median pooled/unpooled std:\n"
    )
    for k, name in enumerate(PARAM_NAMES):
        summary_text += f"  {name:<12} {median_shrink[k]:.3f}\n"
    summary_text += (
        f"\nA factor < 1 means hierarchical\n"
        f"pooling tightened the per-patient\n"
        f"posterior on that parameter.\n"
    )
    ax_f.text(0.05, 0.95, summary_text, transform=ax_f.transAxes,
              fontsize=9, verticalalignment="top", family="monospace",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="lavender", alpha=0.8))

    fig.suptitle(
        "Hierarchical Bayesian fit pooling across the Bruchovsky cohort — closes WP1 §7.2 limitation #1",
        fontsize=12,
    )

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True,
        ).strip()
    except subprocess.CalledProcessError:
        sha = "uncommitted"
    date = dt.date.today().isoformat()
    base = _REPO_ROOT / "results" / "figures" / f"fig22_hierarchical_{cohort_name.lower()}_{sha}_{date}"
    fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {base}.png and .pdf")

    summary = {
        "experiment": f"hierarchical_{cohort_name.lower()}",
        "git_sha": sha,
        "date": date,
        "data_source": "http://www.nicholasbruchovsky.com/dataTanaka.zip",
        "n_patients_loaded": int(cohort.n_patients),
        "n_patients_summaries": int(n_pat),
        "settings": {
            "mh_n_chains": int(n_chains),
            "mh_n_steps": int(n_steps),
            "mh_burn_in": int(burn_in),
            "h_n_chains": int(n_h_chains),
            "h_n_warmup": int(n_h_warmup),
            "h_n_samples": int(n_h_samples),
        },
        "param_names": PARAM_NAMES,
        "population_level_posterior": {
            "mu_mean": mu_mean.tolist(),
            "mu_std": mu_std.tolist(),
            "sigma_pop_mean": sigma_mean.tolist(),
            "sigma_pop_std": sigma_std.tolist(),
            "rhat_mu": h.rhat_pop_mean.tolist(),
            "rhat_sigma": h.rhat_pop_std.tolist(),
            "converged_at_1_10": h.converged(),
        },
        "shrinkage": {
            "median_shrinkage_per_param": shrink["median_shrinkage_per_param"],
            "mean_shrinkage_per_param": shrink["mean_shrinkage_per_param"],
        },
    }
    summary_path = _REPO_ROOT / "results" / f"hierarchical_{cohort_name.lower()}_summary_{sha}_{date}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical Bayes on Bruchovsky cohort")
    parser.add_argument("--n-patients", type=int, default=None,
                        help="Subsample to first N patients (default: all 72)")
    parser.add_argument("--mh-chains", type=int, default=2, dest="n_chains")
    parser.add_argument("--mh-n-steps", type=int, default=600, dest="n_steps")
    parser.add_argument("--mh-burn-in", type=int, default=200, dest="burn_in")
    parser.add_argument("--h-warmup", type=int, default=500, dest="n_h_warmup")
    parser.add_argument("--h-samples", type=int, default=1000, dest="n_h_samples")
    parser.add_argument("--h-chains", type=int, default=4, dest="n_h_chains")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cohort", type=str, default="bruchovsky",
                        choices=["bruchovsky", "shaw"],
                        help="Which IADT cohort to fit (default: bruchovsky)")
    args = parser.parse_args()
    main(
        n_patients=args.n_patients, n_chains=args.n_chains,
        n_steps=args.n_steps, burn_in=args.burn_in,
        n_h_warmup=args.n_h_warmup, n_h_samples=args.n_h_samples,
        n_h_chains=args.n_h_chains, seed=args.seed,
        cohort_name=args.cohort,
    )
