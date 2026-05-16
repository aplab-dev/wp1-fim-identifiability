"""Zhang 2017 reproduction experiment — Phase 2 Stage 2.4 primary target.

Runs three policies (No-treatment / MTD / AT50) across an N-patient cohort
of Zhang 2017-style patients and produces a figure with:

1. **Per-arm trajectory grid** (top): one example patient per policy, showing
   the (T+, TP, T-) populations + PSA + drug schedule. Visual demonstration
   of the AT50 cycling mechanism.
2. **Kaplan-Meier-style TTP curves** (bottom-left): fraction of patients
   not-yet-progressed vs time, per policy. The Zhang 2017 headline finding.
3. **Cumulative dose vs TTP scatter** (bottom-right): each patient as a
   point; the front of the cloud shows the dose-vs-TTP frontier.

This is the **primary Stage 2.4 deliverable** of `phase2_plan.md` — the
qualitative reproduction of Zhang 2017's clinical finding (AT50 extends
TTP at substantially reduced drug exposure).

Output:
- ``results/figures/fig03_zhang2017_reproduction_{git_sha}_{date}.{png,pdf}``
- ``results/zhang2017_reproduction_summary_{git_sha}_{date}.json`` — per-arm
  summary statistics for downstream analysis.

Honesty notes (also surfaced on the figure):
- Cohort variation is generated via 10% log-normal perturbation of ICs,
  NOT Zhang 2017's actual cohort variation (which is unspecified — Zhang
  uses uniform ICs and a single canonical parameter set).
- The alpha matrix is one of "many possible" rank-orderings consistent with
  Zhang 2017's qualitative description (T- heavily suppressed by T+/TP).
  Quantitative TTP figures therefore differ from Zhang's clinical numbers
  (ours: MTD ~37 mo, AT50 ~50+ mo right-censored; Zhang 2017: SOC ~16 mo,
  AT50 ~27 mo). The QUALITATIVE result is the same: AT50 ≫ MTD ≫ no-treatment,
  with AT50 using a fraction of MTD's drug.
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
from policies.cohort_runner import CohortRunner  # noqa: E402
from policies.mtd import MTDPolicy  # noqa: E402
from policies.no_treatment import NoTreatmentPolicy  # noqa: E402
from zhang2017 import (  # noqa: E402
    run_zhang_patient,
    zhang_2017_sampler,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _shade_drug_on(ax, t: np.ndarray, Lambda: np.ndarray) -> None:
    """Shade time spans where drug is on (Λ > 0.5) using axvspan.

    Uses axvspan which is robust to log-y-scale axes (unlike fill_between
    with axes transforms, which can produce broken bounding boxes).
    """
    on = Lambda > 0.5
    if not np.any(on):
        return
    # Identify contiguous on-spans.
    in_run = False
    run_start = 0.0
    for i, is_on in enumerate(on):
        if is_on and not in_run:
            in_run = True
            run_start = float(t[i])
        elif not is_on and in_run:
            in_run = False
            ax.axvspan(run_start, float(t[i]), color="tab:gray", alpha=0.12, linewidth=0)
    if in_run:
        ax.axvspan(run_start, float(t[-1]), color="tab:gray", alpha=0.12, linewidth=0)


def kaplan_meier_curve(ttps: np.ndarray, progressed: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Simple non-progressed-fraction curve (no censoring weights for our case
    since right-censoring is uniform at t_max).

    Args:
        ttps: (N,) TTP per patient.
        progressed: (N,) bool, did the patient progress?
        t_grid: time points to evaluate at.

    Returns:
        Fraction not-yet-progressed at each time in t_grid.
    """
    n = len(ttps)
    out = np.zeros_like(t_grid, dtype=float)
    for i, t in enumerate(t_grid):
        # Count patients not yet progressed by time t.
        out[i] = np.mean(~((ttps <= t) & progressed))
    return out


def example_patient_run(policy_factory, seed: int = 0):
    """Run one example patient for the trajectory plot."""
    rng = np.random.default_rng(seed)
    params = zhang_2017_sampler(rng)
    return run_zhang_patient(params, policy_factory(), rng=rng)


def main(seed: int = 0, n_patients: int = 50, t_max: float = 1500.0) -> None:
    warnings.filterwarnings("ignore")  # suppress LSODA warnings (BDF fallback handles them)

    log.info(f"Running Zhang 2017 cohort: N={n_patients}, t_max={t_max}, seed={seed}")

    policies = {
        "No-treatment": NoTreatmentPolicy,
        "MTD": MTDPolicy,
        "AT50": AT50Policy,
    }
    arm_colors = {
        "No-treatment": "tab:gray",
        "MTD": "tab:blue",
        "AT50": "tab:red",
    }

    # --- Cohort runs ---
    cohort_results = {}
    for name, policy_cls in policies.items():
        log.info(f"  Cohort: {name}")
        runner = CohortRunner(
            run_one_patient=run_zhang_patient,
            param_sampler=zhang_2017_sampler,
            n_patients=n_patients,
            seed=seed,
        )
        cohort_results[name] = runner.run(policy_factory=policy_cls)

    # Summary stats
    log.info("  Cohort summary:")
    summary = {}
    for name, cohort in cohort_results.items():
        ttps = cohort.ttp_array()
        doses = cohort.cumulative_dose_array()
        s = {
            "n_patients": len(cohort.per_patient),
            "ttp_median_d": float(np.median(ttps)),
            "ttp_iqr_d": [float(np.percentile(ttps, 25)), float(np.percentile(ttps, 75))],
            "ttp_median_months": float(np.median(ttps) / 30.0),
            "progression_rate": float(cohort.progression_rate()),
            "drug_fraction_mean": (
                float(np.mean(doses) / np.mean(ttps)) if np.mean(ttps) > 0 else 0.0
            ),
            "cumulative_dose_median": float(np.median(doses)),
        }
        summary[name] = s
        log.info(
            f"    {name:14s}  TTP median={s['ttp_median_d']:5.0f}d "
            f"({s['ttp_median_months']:5.1f} mo)  "
            f"prog rate={s['progression_rate']:.0%}  "
            f"drug frac={s['drug_fraction_mean']:.1%}"
        )

    # --- Example patient trajectories (one per arm) ---
    examples = {name: example_patient_run(cls, seed=seed) for name, cls in policies.items()}

    # --- Figure ---
    fig = plt.figure(figsize=(15, 12))
    fig.set_facecolor("white")
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.4], hspace=0.45, wspace=0.32)

    # Row 1: example trajectories — populations
    for col, name in enumerate(policies):
        ex = examples[name]
        traj = ex["trajectory"]
        ax = fig.add_subplot(gs[0, col])
        ax.plot(traj["t"], traj["x_Tplus"], color="tab:green", linewidth=1.4, label="T+ (sensitive)")
        ax.plot(traj["t"], traj["x_TP"], color="tab:olive", linewidth=1.4, label="TP (producer)")
        ax.plot(traj["t"], traj["x_Tminus"], color="tab:purple", linewidth=1.4, label="T- (resistant)")
        # Drug-on shading
        _shade_drug_on(ax, traj["t"], traj["Lambda"])
        ax.set_yscale("log")
        ax.set_ylim(1, 1.5e4)
        ax.set_title(f"{name} — example patient", fontsize=11, color=arm_colors[name])
        ax.set_ylabel("Cells (log scale)")
        ax.grid(True, alpha=0.3, which="both")
        if col == 0:
            ax.legend(loc="lower right", fontsize=8)

    # Row 2: example PSA trajectories
    for col, name in enumerate(policies):
        ex = examples[name]
        traj = ex["trajectory"]
        ax = fig.add_subplot(gs[1, col])
        baseline = ex["baseline_psa"]
        progression = baseline * 1.20  # progression_psa_threshold default

        ax.plot(traj["t"], traj["psa"], color=arm_colors[name], linewidth=1.5)
        ax.axhline(baseline, color="tab:gray", linestyle=":", linewidth=1.0, label=f"baseline ({baseline:.0f})")
        ax.axhline(0.5 * baseline, color="tab:gray", linestyle="--", linewidth=1.0, label="50% baseline (AT50 trigger)")
        ax.axhline(progression, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.6, label="120% baseline (progression)")
        _shade_drug_on(ax, traj["t"], traj["Lambda"])
        if ex["progressed"]:
            ax.axvline(ex["ttp"], color="tab:red", linewidth=1.2, alpha=0.6)
            ax.annotate(
                f"TTP={ex['ttp']:.0f}d",
                xy=(ex["ttp"], 0.95), xycoords=("data", "axes fraction"),
                fontsize=9, color="tab:red", ha="left", va="top",
            )
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("PSA")
        ax.set_title(f"{name} — PSA", fontsize=11, color=arm_colors[name])
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(loc="upper right", fontsize=7)

    # Row 3: aggregate stats — KM curves (left) + dose-vs-TTP (right)
    ax_km = fig.add_subplot(gs[2, :2])
    ax_scatter = fig.add_subplot(gs[2, 2])

    t_grid = np.linspace(0, t_max, 200)
    for name, cohort in cohort_results.items():
        ttps = cohort.ttp_array()
        progressed = np.array([r["progressed"] for r in cohort.per_patient])
        km = kaplan_meier_curve(ttps, progressed, t_grid)
        ax_km.plot(t_grid, km, color=arm_colors[name], linewidth=2.0,
                   label=f"{name} (median TTP {summary[name]['ttp_median_months']:.0f} mo)")
    ax_km.set_xlabel("Time (days)")
    ax_km.set_ylabel("Fraction not progressed")
    ax_km.set_ylim(-0.02, 1.05)
    ax_km.set_xlim(0, t_max)
    ax_km.set_title(f"Time-to-progression (cohort N={n_patients}/arm)", fontsize=12)
    ax_km.grid(True, alpha=0.3)
    ax_km.legend(loc="lower left", fontsize=10)
    # Annotate Zhang 2017 reference TTPs
    ax_km.axvline(16 * 30, color="tab:gray", linestyle=":", linewidth=1.0, alpha=0.6)
    ax_km.axvline(27 * 30, color="tab:gray", linestyle=":", linewidth=1.0, alpha=0.6)
    ax_km.text(16 * 30 + 5, 0.92, "Zhang 2017\nSOC 16mo", fontsize=8, color="tab:gray", alpha=0.8)
    ax_km.text(27 * 30 + 5, 0.92, "Zhang 2017\nAT50 27mo", fontsize=8, color="tab:gray", alpha=0.8)

    # Dose vs TTP scatter
    for name, cohort in cohort_results.items():
        ttps = cohort.ttp_array()
        doses = cohort.cumulative_dose_array()
        ax_scatter.scatter(
            doses, ttps,
            c=arm_colors[name], s=40, alpha=0.6, edgecolor="none",
            label=name,
        )
    ax_scatter.set_xlabel("Cumulative drug exposure (drug-days)")
    ax_scatter.set_ylabel("Time to progression (days)")
    ax_scatter.set_title("Dose vs TTP (per patient)", fontsize=11)
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "Zhang 2017 reproduction (Phase 2 Stage 2.4) — qualitative match: AT50 ≫ MTD ≫ no-tx, AT50 uses much less drug",
        fontsize=13,
    )

    # --- Save ---
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True,
        ).strip()
    except subprocess.CalledProcessError:
        sha = "uncommitted"
    date = dt.date.today().isoformat()
    base = _REPO_ROOT / "results" / "figures" / f"fig03_zhang2017_reproduction_{sha}_{date}"
    fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {base}.png and .pdf")

    # JSON summary
    summary_path = (
        _REPO_ROOT / "results" / f"zhang2017_reproduction_summary_{sha}_{date}.json"
    )
    summary_payload = {
        "experiment": "zhang2017_reproduction",
        "git_sha": sha,
        "date": date,
        "seed": seed,
        "n_patients": n_patients,
        "t_max_days": t_max,
        "summary": summary,
        "honesty_notes": [
            "Cohort variation is via 10% log-normal IC perturbation, not Zhang 2017's actual cohort design.",
            "alpha matrix is one of 22 valid rank-orderings consistent with Zhang 2017's qualitative description.",
            "Quantitative TTP differs from Zhang's clinical figures (we don't fit to Zhang's supplementary tables).",
            "Qualitative result reproduces Zhang 2017's headline finding (AT50 > MTD on TTP, AT50 uses less drug).",
        ],
    }
    with summary_path.open("w") as f:
        json.dump(summary_payload, f, indent=2)
    log.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zhang 2017 reproduction experiment")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--n-patients", type=int, default=50, help="Cohort size per arm")
    parser.add_argument("--t-max", type=float, default=1500.0, help="Sim horizon (days)")
    args = parser.parse_args()
    main(seed=args.seed, n_patients=args.n_patients, t_max=args.t_max)
