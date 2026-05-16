"""Cross-cohort PA-vs-PE comparison: Bruchovsky et al. + Shaw et al.

Validates the experiment 19 finding (37% PE-vs-PA disagreement on Bruchovsky)
by re-running the full pipeline on the independent Shaw et al. 2007 cohort
(also in the dataTanaka archive).

If the Shaw disagreement rate is comparable to Bruchovsky's, the WP4 main
empirical claim is robust to cohort choice — this is the single most
important external validation we can do without acquiring a third dataset.

Output:
- ``results/figures/fig20_cross_cohort_pa_vs_pe_{git_sha}_{date}.{png,pdf}``
- ``results/cross_cohort_pa_vs_pe_summary_{git_sha}_{date}.json``
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

# Import the per-patient evaluator from experiment 19 (lift the function).
sys.path.insert(0, str(_REPO_ROOT / "src" / "experiments"))

from realdata import load_dataTanaka, load_shaw_et_al  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def evaluate_cohort(cohort, label: str, n_steps: int, burn_in: int,
                    n_post_samples: int, n_sim_per: int, seed: int) -> list[dict]:
    """Run experiment-19's per-patient evaluator on a cohort. Returns list of result dicts."""
    # Lift evaluate_one_patient from experiment 19
    import importlib.util
    exp_path = _REPO_ROOT / "src" / "experiments" / "19_real_cohort_pa_vs_pe.py"
    spec = importlib.util.spec_from_file_location("exp19", exp_path)
    exp19 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp19)

    log.info(f"  Evaluating {label} cohort (n={cohort.n_patients})")
    results = []
    for i, patient in enumerate(cohort.patients):
        try:
            res = exp19.evaluate_one_patient(
                patient, n_chains=2, n_steps=n_steps, burn_in=burn_in,
                n_post_samples=n_post_samples, n_sim_per=n_sim_per,
                seed=seed * 1000 + i,
            )
            if res is None:
                continue
            res["cohort"] = label
            results.append(res)
            if (i + 1) % 5 == 0:
                log.info(f"    {label} [{i+1}/{cohort.n_patients}]")
        except Exception as e:  # noqa: BLE001
            log.warning(f"    {label} patient {i}: {e}")
            continue
    return results


def cohort_aggregate(results: list[dict], label: str) -> dict:
    if not results:
        return {"label": label, "n": 0}
    n = len(results)
    n_pe_at = sum(1 for r in results if r["pe_choice"] == "AT50")
    n_pa_at = sum(1 for r in results if r["pa_choice"] == "AT50")
    n_disagree = sum(1 for r in results if r["pe_pa_disagree"])
    p_at = np.array([r["pa_p_at50_wins"] for r in results])
    n_sensitive = int(np.sum((p_at > 0.10) & (p_at < 0.90)))
    n_converged_loose = sum(1 for r in results if r["mcmc_converged_at_1_50"])
    return {
        "label": label,
        "n": n,
        "pe_at50_count": int(n_pe_at),
        "pe_at50_rate": n_pe_at / n,
        "pa_at50_count": int(n_pa_at),
        "pa_at50_rate": n_pa_at / n,
        "disagreement_count": int(n_disagree),
        "disagreement_rate": n_disagree / n,
        "posterior_sensitive_count": int(n_sensitive),
        "posterior_sensitive_rate": n_sensitive / n,
        "mcmc_converged_loose_count": int(n_converged_loose),
        "p_at50_wins_distribution_quartiles": [
            float(np.percentile(p_at, 25)),
            float(np.percentile(p_at, 50)),
            float(np.percentile(p_at, 75)),
        ],
    }


def main(n_steps: int = 400, burn_in: int = 150, n_post_samples: int = 12,
         n_sim_per: int = 2, seed: int = 0) -> None:
    warnings.filterwarnings("ignore")
    log.info("Cross-cohort PA-vs-PE comparison: Bruchovsky_et_al + Shaw_et_al")

    bruchovsky = load_dataTanaka()
    shaw = load_shaw_et_al()
    log.info(f"  Bruchovsky: {bruchovsky.n_patients} patients (clinical prog={bruchovsky.progression_rate():.0%})")
    log.info(f"  Shaw:       {shaw.n_patients} patients (clinical prog={shaw.progression_rate():.0%})")

    bruch_results = evaluate_cohort(
        bruchovsky, "Bruchovsky", n_steps, burn_in, n_post_samples, n_sim_per, seed,
    )
    shaw_results = evaluate_cohort(
        shaw, "Shaw", n_steps, burn_in, n_post_samples, n_sim_per, seed + 7,
    )

    bruch_agg = cohort_aggregate(bruch_results, "Bruchovsky")
    shaw_agg = cohort_aggregate(shaw_results, "Shaw")

    log.info("Cross-cohort summary:")
    for agg in [bruch_agg, shaw_agg]:
        if agg["n"] == 0:
            log.warning(f"  {agg['label']}: NO RESULTS")
            continue
        log.info(
            f"  {agg['label']:11s}  n={agg['n']:3d}  "
            f"PE-AT50={agg['pe_at50_rate']:.0%}  PA-AT50={agg['pa_at50_rate']:.0%}  "
            f"disagree={agg['disagreement_rate']:.0%}  sensitive={agg['posterior_sensitive_rate']:.0%}"
        )

    # --- Figure ---
    fig = plt.figure(figsize=(14, 9))
    fig.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

    # Panel A: PE/PA AT50 rate per cohort
    ax_a = fig.add_subplot(gs[0, 0])
    cohorts = ["Bruchovsky", "Shaw"]
    pe_rates = [bruch_agg.get("pe_at50_rate", 0), shaw_agg.get("pe_at50_rate", 0)]
    pa_rates = [bruch_agg.get("pa_at50_rate", 0), shaw_agg.get("pa_at50_rate", 0)]
    x = np.arange(len(cohorts))
    width = 0.4
    ax_a.bar(x - width / 2, pe_rates, width, color="tab:blue", label="PE recommends AT50", alpha=0.8)
    ax_a.bar(x + width / 2, pa_rates, width, color="tab:red", label="PA recommends AT50", alpha=0.8)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(cohorts)
    ax_a.set_ylabel("Fraction of cohort")
    ax_a.set_title("[A] AT50-recommendation rate (PE vs PA)", fontsize=10)
    ax_a.legend(fontsize=8)
    ax_a.grid(True, alpha=0.3, axis="y")
    ax_a.set_ylim(0, 1)

    # Panel B: disagreement rate per cohort (the headline)
    ax_b = fig.add_subplot(gs[0, 1])
    disagree_rates = [bruch_agg.get("disagreement_rate", 0), shaw_agg.get("disagreement_rate", 0)]
    bars = ax_b.bar(cohorts, disagree_rates, color=["tab:purple", "tab:orange"], alpha=0.8, edgecolor="black")
    for bar, rate, agg in zip(bars, disagree_rates, [bruch_agg, shaw_agg]):
        ax_b.text(bar.get_x() + bar.get_width() / 2, rate + 0.01,
                  f"{rate:.0%}\n({agg.get('disagreement_count', 0)}/{agg.get('n', 0)})",
                  ha="center", va="bottom", fontsize=10)
    ax_b.set_ylabel("PE-vs-PA disagreement rate")
    ax_b.set_title("[B] Disagreement rate (cross-cohort consistency check)", fontsize=10)
    ax_b.grid(True, alpha=0.3, axis="y")
    ax_b.set_ylim(0, max(0.6, max(disagree_rates) + 0.1) if disagree_rates else 0.6)

    # Panel C: posterior-sensitive rate per cohort
    ax_c = fig.add_subplot(gs[0, 2])
    ps_rates = [bruch_agg.get("posterior_sensitive_rate", 0), shaw_agg.get("posterior_sensitive_rate", 0)]
    ax_c.bar(cohorts, ps_rates, color=["tab:purple", "tab:orange"], alpha=0.5, edgecolor="black")
    ax_c.set_ylabel("Fraction posterior-sensitive")
    ax_c.set_title("[C] Posterior-sensitive (10% < P(AT50) < 90%)", fontsize=10)
    ax_c.grid(True, alpha=0.3, axis="y")

    # Panel D: P(AT50 wins) distribution overlay
    ax_d = fig.add_subplot(gs[1, 0])
    bruch_p = np.array([r["pa_p_at50_wins"] for r in bruch_results]) if bruch_results else np.array([])
    shaw_p = np.array([r["pa_p_at50_wins"] for r in shaw_results]) if shaw_results else np.array([])
    bins = np.linspace(0, 1, 21)
    if len(bruch_p) > 0:
        ax_d.hist(bruch_p, bins=bins, alpha=0.5, color="tab:purple", label=f"Bruchovsky (n={len(bruch_p)})")
    if len(shaw_p) > 0:
        ax_d.hist(shaw_p, bins=bins, alpha=0.5, color="tab:orange", label=f"Shaw (n={len(shaw_p)})")
    ax_d.axvline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax_d.set_xlabel("P(AT50 beats MTD)")
    ax_d.set_ylabel("Patient count")
    ax_d.set_title("[D] Per-patient P(AT50 wins TTP)", fontsize=10)
    ax_d.legend(fontsize=8)
    ax_d.grid(True, alpha=0.3)

    # Panel E: PE vs PA TTP advantage scatter
    ax_e = fig.add_subplot(gs[1, 1])
    for results, color, label in [(bruch_results, "tab:purple", "Bruchovsky"),
                                  (shaw_results, "tab:orange", "Shaw")]:
        if not results:
            continue
        pe_adv = np.array([r["pe_ttp_at50_d"] - r["pe_ttp_mtd_d"] for r in results]) / 30
        pa_adv = np.array([r["pa_expected_ttp_at50_d"] - r["pa_expected_ttp_mtd_d"] for r in results]) / 30
        disagree = np.array([r["pe_pa_disagree"] for r in results])
        ax_e.scatter(pe_adv[~disagree], pa_adv[~disagree], c=color, s=30, alpha=0.4,
                     edgecolor="none", label=f"{label} (PE=PA)")
        ax_e.scatter(pe_adv[disagree], pa_adv[disagree], c=color, s=60, alpha=0.9,
                     edgecolor="black", label=f"{label} (disagree)")
    diag = np.linspace(-50, 50, 100)
    ax_e.plot(diag, diag, "k--", alpha=0.4, linewidth=0.7)
    ax_e.axhline(0, color="black", linewidth=0.5)
    ax_e.axvline(0, color="black", linewidth=0.5)
    ax_e.set_xlabel("PE AT50 advantage (months)")
    ax_e.set_ylabel("PA AT50 advantage (months)")
    ax_e.set_title("[E] PE vs PA per-patient (both cohorts)", fontsize=10)
    ax_e.legend(fontsize=7)
    ax_e.grid(True, alpha=0.3)

    # Panel F: summary text
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")
    summary_text = (
        f"CROSS-COHORT VALIDATION\n\n"
        f"Source: nicholasbruchovsky.com/dataTanaka.zip\n\n"
        f"Bruchovsky et al. (n={bruch_agg.get('n', 0)}):\n"
        f"  PE-AT50: {bruch_agg.get('pe_at50_rate', 0):.0%}\n"
        f"  PA-AT50: {bruch_agg.get('pa_at50_rate', 0):.0%}\n"
        f"  Disagree: {bruch_agg.get('disagreement_rate', 0):.0%}\n"
        f"  Sensitive: {bruch_agg.get('posterior_sensitive_rate', 0):.0%}\n\n"
        f"Shaw et al. (n={shaw_agg.get('n', 0)}):\n"
        f"  PE-AT50: {shaw_agg.get('pe_at50_rate', 0):.0%}\n"
        f"  PA-AT50: {shaw_agg.get('pa_at50_rate', 0):.0%}\n"
        f"  Disagree: {shaw_agg.get('disagreement_rate', 0):.0%}\n"
        f"  Sensitive: {shaw_agg.get('posterior_sensitive_rate', 0):.0%}\n\n"
        f"WP4 main empirical: cross-cohort consistency.\n"
        f"If both cohorts show comparable rates,\n"
        f"the methodology message is robust."
    )
    ax_f.text(0.05, 0.95, summary_text, transform=ax_f.transAxes,
             fontsize=9, verticalalignment="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    fig.suptitle(
        f"Cross-cohort PA-vs-PE: Bruchovsky vs Shaw (real IADT trials)",
        fontsize=12,
    )

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True,
        ).strip()
    except subprocess.CalledProcessError:
        sha = "uncommitted"
    date = dt.date.today().isoformat()
    base = _REPO_ROOT / "results" / "figures" / f"fig20_cross_cohort_pa_vs_pe_{sha}_{date}"
    fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {base}.png and .pdf")

    summary = {
        "experiment": "cross_cohort_pa_vs_pe",
        "git_sha": sha,
        "date": date,
        "data_source": "http://www.nicholasbruchovsky.com/dataTanaka.zip",
        "settings": {
            "n_chains": 2, "n_steps": int(n_steps), "burn_in": int(burn_in),
            "n_post_samples": int(n_post_samples), "n_sim_per": int(n_sim_per),
        },
        "cohorts": {
            "Bruchovsky": bruch_agg,
            "Shaw": shaw_agg,
        },
        "per_patient": {
            "Bruchovsky": bruch_results,
            "Shaw": shaw_results,
        },
    }
    summary_path = _REPO_ROOT / "results" / f"cross_cohort_pa_vs_pe_summary_{sha}_{date}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-cohort PA-vs-PE comparison")
    parser.add_argument("--mh-n-steps", type=int, default=400, dest="n_steps")
    parser.add_argument("--burn-in", type=int, default=150)
    parser.add_argument("--n-post-samples", type=int, default=12)
    parser.add_argument("--n-sim-per", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(n_steps=args.n_steps, burn_in=args.burn_in,
         n_post_samples=args.n_post_samples, n_sim_per=args.n_sim_per,
         seed=args.seed)
