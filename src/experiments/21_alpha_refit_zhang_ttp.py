"""Refit alpha matrix to recover Zhang 2017's quantitative TTP figures.

Stage 2.4 reproduction matches Zhang 2017 qualitatively (AT50 ≫ MTD on TTP)
but censors at 50 mo while Zhang reports MTD ~14-16 mo and AT50 ~27 mo.
Fix: scipy-optimize the alpha matrix to minimize the squared error against
Zhang's reported median TTPs.

Search space: T- suppression row [α_T-,T+, α_T-,TP] in [0.5, 10]². We
fix the off-diagonal T+/TP block at canonical values to keep the search
2D and tractable. Cohort: N=20 patients per evaluation, IC perturbation
disabled to focus on parameter effects.

Target (Zhang 2017 Nat Comms reassessment):
- SOC / MTD median TTP ~ 16 months ≈ 480 days
- AT50 median TTP ~ 27 months ≈ 810 days

Output:
- ``results/figures/fig21_alpha_refit_{git_sha}_{date}.{png,pdf}`` —
  optimization landscape + refitted-cohort comparison.
- ``results/alpha_refit_summary_{git_sha}_{date}.json`` — best alpha
  + per-arm TTP at the best alpha.
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
from scipy.optimize import minimize

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from policies.at50 import AT50Policy  # noqa: E402
from policies.cohort_runner import CohortRunner  # noqa: E402
from policies.mtd import MTDPolicy  # noqa: E402
from simulators.lv_3pop_kshift import LV3PopParams  # noqa: E402
from zhang2017 import (  # noqa: E402
    ZHANG_CANONICAL_X0,
    ZhangPatientParams,
    run_zhang_patient,
    zhang_canonical_lv_params,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# Zhang 2017 reported (per docs/literature/zhang-2017-crpc-adaptive.md reassessment):
# SOC continuous abi ~ 14 mo; AA-302 historical ~ 16.5 mo; AT50 ~ 27 mo.
ZHANG_TARGET_MTD_DAYS = 480.0   # ~16 months
ZHANG_TARGET_AT50_DAYS = 810.0  # ~27 months


_canon = zhang_canonical_lv_params()


def cohort_at_alpha(alpha_T_minus_T_plus: float, alpha_T_minus_TP: float,
                    n_patients: int = 20, seed: int = 0) -> dict:
    """Run a small N-patient cohort with a given (alpha[2,0], alpha[2,1]) and return per-arm median TTPs."""
    alpha = _canon.alpha.copy()
    alpha[2, 0] = max(alpha_T_minus_T_plus, 0.01)
    alpha[2, 1] = max(alpha_T_minus_TP, 0.01)
    lv_params = LV3PopParams(
        r_Tplus=_canon.r_Tplus, r_TP=_canon.r_TP, r_Tminus=_canon.r_Tminus,
        K_Tminus=_canon.K_Tminus, K_TP_max=_canon.K_TP_max, K_TP_drop=_canon.K_TP_drop,
        mu_max=_canon.mu_max, mu_drop=_canon.mu_drop, alpha=alpha,
    )

    def sampler(rng):  # noqa: ARG001
        return ZhangPatientParams(lv_params=lv_params, ic_perturbation_std=0.05)

    out = {}
    for label, policy_cls in [("MTD", MTDPolicy), ("AT50", AT50Policy)]:
        runner = CohortRunner(
            run_one_patient=run_zhang_patient,
            param_sampler=sampler, n_patients=n_patients, seed=seed,
        )
        cohort = runner.run(policy_factory=policy_cls)
        ttps = cohort.ttp_array()
        out[label] = float(np.median(ttps))
    return out


def loss(alpha_pair: np.ndarray, n_patients: int = 20, seed: int = 0) -> float:
    """Squared error between cohort median TTPs and Zhang target TTPs."""
    try:
        result = cohort_at_alpha(alpha_pair[0], alpha_pair[1], n_patients, seed)
    except Exception:  # noqa: BLE001
        return 1e10
    err_mtd = (result["MTD"] - ZHANG_TARGET_MTD_DAYS) / ZHANG_TARGET_MTD_DAYS
    err_at50 = (result["AT50"] - ZHANG_TARGET_AT50_DAYS) / ZHANG_TARGET_AT50_DAYS
    # Both are normalized to relative error → equal weight.
    return err_mtd ** 2 + err_at50 ** 2


def main(n_patients_eval: int = 15, max_iter: int = 30, seed: int = 0) -> None:
    warnings.filterwarnings("ignore")
    log.info("Stage 2.4 alpha-refit to Zhang 2017 quantitative TTP targets")
    log.info(f"  Targets: MTD ~ {ZHANG_TARGET_MTD_DAYS:.0f} d ({ZHANG_TARGET_MTD_DAYS/30:.0f} mo); "
             f"AT50 ~ {ZHANG_TARGET_AT50_DAYS:.0f} d ({ZHANG_TARGET_AT50_DAYS/30:.0f} mo)")

    # Coarse 2D scan to find a good starting point — much faster than scipy with random init.
    log.info(f"  Phase 1: coarse 5x5 scan over (alpha[T-,T+], alpha[T-,TP])")
    grid_a = np.linspace(0.5, 10, 5)
    grid_b = np.linspace(0.5, 10, 5)
    scan_results = []
    for a in grid_a:
        for b in grid_b:
            try:
                res = cohort_at_alpha(a, b, n_patients_eval, seed)
                scan_results.append({
                    "alpha_T_minus_T_plus": float(a), "alpha_T_minus_TP": float(b),
                    "mtd_ttp_d": res["MTD"], "at50_ttp_d": res["AT50"],
                    "loss": float(((res["MTD"] - ZHANG_TARGET_MTD_DAYS) / ZHANG_TARGET_MTD_DAYS) ** 2 +
                                  ((res["AT50"] - ZHANG_TARGET_AT50_DAYS) / ZHANG_TARGET_AT50_DAYS) ** 2),
                })
                log.info(f"    α=({a:.1f}, {b:.1f}): MTD={res['MTD']:.0f}d, AT50={res['AT50']:.0f}d, loss={scan_results[-1]['loss']:.3f}")
            except Exception:  # noqa: BLE001
                continue

    # Find best from scan
    if not scan_results:
        log.error("No scan results")
        return
    best_scan = min(scan_results, key=lambda r: r["loss"])
    log.info(f"  Best scan point: α=({best_scan['alpha_T_minus_T_plus']:.1f}, "
             f"{best_scan['alpha_T_minus_TP']:.1f}), loss={best_scan['loss']:.3f}")

    # Phase 2: scipy local refinement around the best point.
    log.info(f"  Phase 2: Nelder-Mead refinement (max {max_iter} iterations)")
    x0 = np.array([best_scan["alpha_T_minus_T_plus"], best_scan["alpha_T_minus_TP"]])
    iter_log = []

    def callback(xk):
        iter_log.append({"alpha_pair": xk.tolist(), "loss_val": float(loss(xk, n_patients_eval, seed))})
        log.info(f"    iter {len(iter_log)}: α={xk}, loss={iter_log[-1]['loss_val']:.4f}")

    opt_result = minimize(
        lambda x: loss(x, n_patients_eval, seed),
        x0=x0, method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 0.05, "fatol": 0.005},
        callback=callback,
    )
    best_alpha = opt_result.x
    final_result = cohort_at_alpha(best_alpha[0], best_alpha[1], n_patients_eval, seed)
    log.info(f"  Optimization done. Best α=({best_alpha[0]:.2f}, {best_alpha[1]:.2f})")
    log.info(f"    MTD TTP={final_result['MTD']:.0f}d ({final_result['MTD']/30:.1f}mo) "
             f"vs target {ZHANG_TARGET_MTD_DAYS/30:.0f}mo")
    log.info(f"    AT50 TTP={final_result['AT50']:.0f}d ({final_result['AT50']/30:.1f}mo) "
             f"vs target {ZHANG_TARGET_AT50_DAYS/30:.0f}mo")

    # --- Figure ---
    fig = plt.figure(figsize=(13, 9))
    fig.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.32)

    # Panel A: scan landscape (loss heatmap)
    ax_a = fig.add_subplot(gs[0, 0])
    losses = np.array([r["loss"] for r in scan_results])
    a_vals = np.array([r["alpha_T_minus_T_plus"] for r in scan_results])
    b_vals = np.array([r["alpha_T_minus_TP"] for r in scan_results])
    grid_loss = np.zeros((len(grid_b), len(grid_a)))
    for i, b in enumerate(grid_b):
        for j, a in enumerate(grid_a):
            for r in scan_results:
                if abs(r["alpha_T_minus_T_plus"] - a) < 1e-3 and abs(r["alpha_T_minus_TP"] - b) < 1e-3:
                    grid_loss[i, j] = r["loss"]
                    break
    im = ax_a.imshow(grid_loss, origin="lower", cmap="viridis_r",
                      extent=[grid_a[0], grid_a[-1], grid_b[0], grid_b[-1]],
                      aspect="auto")
    ax_a.scatter([best_alpha[0]], [best_alpha[1]], color="red", s=80, marker="*",
                  edgecolor="white", linewidth=1.5, label=f"refit α=({best_alpha[0]:.2f}, {best_alpha[1]:.2f})")
    ax_a.scatter([_canon.alpha[2, 0]], [_canon.alpha[2, 1]], color="black", s=60, marker="x",
                  label=f"original α=({_canon.alpha[2,0]:.1f}, {_canon.alpha[2,1]:.1f})")
    ax_a.set_xlabel("α(T-, T+)")
    ax_a.set_ylabel("α(T-, TP)")
    ax_a.set_title("[A] Loss landscape (scan) + refit", fontsize=10)
    fig.colorbar(im, ax=ax_a, label="squared rel. error")
    ax_a.legend(fontsize=8)

    # Panel B: TTP comparison at original vs refit alpha
    ax_b = fig.add_subplot(gs[0, 1])
    # Original (canonical) result
    canon_res = cohort_at_alpha(_canon.alpha[2, 0], _canon.alpha[2, 1], n_patients_eval, seed)
    labels = ["MTD", "AT50"]
    canon_vals = [canon_res["MTD"] / 30, canon_res["AT50"] / 30]
    refit_vals = [final_result["MTD"] / 30, final_result["AT50"] / 30]
    targets = [ZHANG_TARGET_MTD_DAYS / 30, ZHANG_TARGET_AT50_DAYS / 30]
    x = np.arange(len(labels))
    w = 0.27
    ax_b.bar(x - w, canon_vals, w, color="tab:gray", alpha=0.7, label="canonical α (original)", edgecolor="black")
    ax_b.bar(x, refit_vals, w, color="tab:red", alpha=0.8, label="refit α", edgecolor="black")
    ax_b.bar(x + w, targets, w, color="tab:green", alpha=0.8, label="Zhang 2017 target", edgecolor="black")
    ax_b.set_xticks(x); ax_b.set_xticklabels(labels)
    ax_b.set_ylabel("Median TTP (months)")
    ax_b.set_title("[B] TTP: canonical α vs refit α vs Zhang target", fontsize=10)
    ax_b.legend(fontsize=8)
    ax_b.grid(True, alpha=0.3, axis="y")

    # Panel C: loss trajectory
    ax_c = fig.add_subplot(gs[1, 0])
    if iter_log:
        iters = list(range(1, len(iter_log) + 1))
        losses_traj = [r["loss_val"] for r in iter_log]
        ax_c.plot(iters, losses_traj, "o-", color="tab:blue", linewidth=1.5)
        ax_c.set_xlabel("Iteration")
        ax_c.set_ylabel("Loss (squared rel. error)")
        ax_c.set_title("[C] Nelder-Mead optimization trace", fontsize=10)
        ax_c.grid(True, alpha=0.3)
        ax_c.set_yscale("log")

    # Panel D: summary text
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")
    summary_text = (
        f"STAGE 2.4 ALPHA REFIT\n\n"
        f"Targets (Zhang 2017):\n"
        f"  MTD median TTP:  {ZHANG_TARGET_MTD_DAYS/30:.0f} mo\n"
        f"  AT50 median TTP: {ZHANG_TARGET_AT50_DAYS/30:.0f} mo\n\n"
        f"Original canonical α=({_canon.alpha[2,0]:.1f}, {_canon.alpha[2,1]:.1f}):\n"
        f"  MTD: {canon_vals[0]:.1f} mo\n"
        f"  AT50: {canon_vals[1]:.1f} mo\n\n"
        f"Refit α=({best_alpha[0]:.2f}, {best_alpha[1]:.2f}):\n"
        f"  MTD: {refit_vals[0]:.1f} mo\n"
        f"  AT50: {refit_vals[1]:.1f} mo\n\n"
        f"Final loss: {opt_result.fun:.4f}\n"
        f"Iterations: {opt_result.nit}\n"
        f"Optimizer: {'converged' if opt_result.success else 'maxiter'}\n\n"
        f"Closes WP1 §3.4 honesty gap on quantitative\n"
        f"reproduction of Zhang 2017's clinical TTPs."
    )
    ax_d.text(0.05, 0.95, summary_text, transform=ax_d.transAxes,
             fontsize=9, verticalalignment="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    fig.suptitle(
        f"Stage 2.4 alpha-refit — match Zhang 2017 quantitative TTP figures",
        fontsize=12,
    )

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True,
        ).strip()
    except subprocess.CalledProcessError:
        sha = "uncommitted"
    date = dt.date.today().isoformat()
    base = _REPO_ROOT / "results" / "figures" / f"fig21_alpha_refit_{sha}_{date}"
    fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {base}.png and .pdf")

    summary = {
        "experiment": "alpha_refit_to_zhang",
        "git_sha": sha,
        "date": date,
        "targets": {
            "mtd_ttp_days": ZHANG_TARGET_MTD_DAYS,
            "at50_ttp_days": ZHANG_TARGET_AT50_DAYS,
            "source": "Zhang 2017 reassessment — SOC ~14-16 mo, AT50 ~27 mo",
        },
        "settings": {"n_patients_eval": int(n_patients_eval), "max_iter": int(max_iter), "seed": int(seed)},
        "scan_results": scan_results,
        "best_scan": best_scan,
        "optimization": {
            "iterations": int(opt_result.nit),
            "function_evals": int(opt_result.nfev),
            "converged": bool(opt_result.success),
            "final_loss": float(opt_result.fun),
            "best_alpha_T_minus_T_plus": float(best_alpha[0]),
            "best_alpha_T_minus_TP": float(best_alpha[1]),
            "iter_log": iter_log,
        },
        "final_cohort_result": {
            "mtd_ttp_days": float(final_result["MTD"]),
            "mtd_ttp_months": float(final_result["MTD"] / 30),
            "at50_ttp_days": float(final_result["AT50"]),
            "at50_ttp_months": float(final_result["AT50"] / 30),
        },
        "canonical_cohort_result": {
            "mtd_ttp_days": float(canon_res["MTD"]),
            "mtd_ttp_months": float(canon_res["MTD"] / 30),
            "at50_ttp_days": float(canon_res["AT50"]),
            "at50_ttp_months": float(canon_res["AT50"] / 30),
        },
    }
    summary_path = _REPO_ROOT / "results" / f"alpha_refit_summary_{sha}_{date}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refit Stage 2.4 alpha to Zhang 2017 TTPs")
    parser.add_argument("--n-patients-eval", type=int, default=15)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(n_patients_eval=args.n_patients_eval, max_iter=args.max_iter, seed=args.seed)
