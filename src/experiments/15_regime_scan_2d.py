"""2D regime scan — M6 follow-up to experiment 13.

Experiment 13 found a posterior-sensitive regime at K_TP_drop = 1000 along
a 1D scan with all other parameters held canonical. This experiment
extends to a 2D scan: K_TP_drop × alpha[T-,T+]. Goal: characterize the
regime-boundary as a hypersurface and visualize where posterior-aware
control matters most.

Each grid point: compute FIM, sample N=15 posterior draws, run cohort
comparison (n=2 patients per draw), compute P(AT50 wins TTP).

Smaller N_posterior and n_patients than experiment 13 because we have
~25 grid points to evaluate. Total: ~1500 simulations. ~5 minutes.

Output:
- ``results/figures/fig15_regime_scan_2d_{git_sha}_{date}.{png,pdf}`` —
  heatmap of P(AT50 wins TTP) over (K_TP_drop, alpha[T-,T+]). Plus
  median-advantage heatmap for sign + magnitude.
- ``results/regime_scan_2d_summary_{git_sha}_{date}.json``.
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

from identifiability import compute_fim  # noqa: E402
from policies.at50 import AT50Policy  # noqa: E402
from policies.mtd import MTDPolicy  # noqa: E402
from simulators.lv_3pop_kshift import LV3PopKShift, LV3PopParams  # noqa: E402
from simulators.psa_dynamics import PSAParams, psa_steady_state  # noqa: E402
from zhang2017 import (  # noqa: E402
    ZHANG_CANONICAL_X0,
    ZhangPatientParams,
    run_zhang_patient,
    zhang_canonical_lv_params,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


PARAM_NAMES = ["r_T+", "r_TP", "r_T-", "α(T-,T+)", "α(T-,TP)", "K_TP_drop"]
_canon = zhang_canonical_lv_params()
_NOMINAL = np.array([
    _canon.r_Tplus, _canon.r_TP, _canon.r_Tminus,
    float(_canon.alpha[2, 0]), float(_canon.alpha[2, 1]),
    _canon.K_TP_drop,
])
T_OBS = np.arange(0.0, 1500.0 + 1, 28.0)


def _build_lv_params(theta: np.ndarray) -> LV3PopParams:
    r_Tplus, r_TP, r_Tminus, alpha_2_0, alpha_2_1, K_TP_drop = theta
    alpha = _canon.alpha.copy()
    alpha[2, 0] = alpha_2_0
    alpha[2, 1] = alpha_2_1
    return LV3PopParams(
        r_Tplus=max(r_Tplus, 1e-6), r_TP=max(r_TP, 1e-6), r_Tminus=max(r_Tminus, 1e-6),
        K_Tminus=_canon.K_Tminus, K_TP_max=_canon.K_TP_max,
        K_TP_drop=max(min(K_TP_drop, _canon.K_TP_max - 1), 1.0),
        mu_max=_canon.mu_max, mu_drop=_canon.mu_drop,
        alpha=alpha,
    )


def predict_psa_under_mtd(theta: np.ndarray) -> np.ndarray | None:
    sim = LV3PopKShift(_build_lv_params(theta))
    psa_params = PSAParams()

    def rhs(t, y):
        x = y[:3]
        psa = y[3]
        dx = sim.dynamics(t, x, Lambda=1.0)
        dpsa = psa_params.rho * float(np.sum(x)) - psa_params.phi * psa
        return np.concatenate([dx, [dpsa]])

    psa0 = psa_steady_state(float(np.sum(ZHANG_CANONICAL_X0)), psa_params)
    y0 = np.array([*ZHANG_CANONICAL_X0, psa0])
    for method in ("LSODA", "BDF"):
        try:
            sol = solve_ivp(rhs, t_span=(0.0, T_OBS[-1]), y0=y0, t_eval=T_OBS,
                            method=method, rtol=1e-8, atol=1e-3)
            if sol.success:
                return sol.y[3]
        except Exception:  # noqa: BLE001
            continue
    return None


def is_physically_valid(theta: np.ndarray) -> bool:
    return (
        theta[0] > 0 and theta[1] > 0 and theta[2] > 0
        and theta[3] >= 0 and theta[4] >= 0
        and 0 < theta[5] < 9999.0
    )


def evaluate_grid_point(theta_center: np.ndarray, n_posterior: int, n_patients: int,
                        rng: np.random.Generator) -> dict | None:
    psa_nom = predict_psa_under_mtd(theta_center)
    if psa_nom is None:
        return None
    sigma = 0.10 * np.maximum(psa_nom, 0.1 * psa_nom.max())
    fim_result = compute_fim(
        predict=predict_psa_under_mtd, theta_nominal=theta_center,
        eps_rel=1e-3, sigma=sigma, param_names=PARAM_NAMES,
    )
    eigvals, eigvecs = np.linalg.eigh(fim_result.fim)
    eigvals_reg = np.maximum(eigvals, eigvals.max() * 1e-3)
    cov = eigvecs @ np.diag(1.0 / eigvals_reg) @ eigvecs.T
    cov = 0.5 * (cov + cov.T)
    L = np.linalg.cholesky(cov + 1e-8 * np.eye(len(theta_center)))
    raw_draws = theta_center[None, :] + (rng.normal(size=(n_posterior, 6)) @ L.T)
    valid_draws = [d for d in raw_draws if is_physically_valid(d)]
    if not valid_draws:
        return None

    advantages = []
    drug_savings = []
    for draw in valid_draws:
        try:
            params_for_runner = ZhangPatientParams(lv_params=_build_lv_params(draw))
            ttp_mtd, ttp_at, dose_mtd, dose_at = [], [], [], []
            ss = np.random.SeedSequence(int(rng.integers(0, 1_000_000)))
            for child in ss.spawn(n_patients):
                patient_rng = np.random.Generator(np.random.PCG64(child))
                try:
                    r_mtd = run_zhang_patient(params_for_runner, MTDPolicy(), rng=patient_rng)
                    r_at = run_zhang_patient(params_for_runner, AT50Policy(), rng=patient_rng)
                    ttp_mtd.append(r_mtd["ttp"]); ttp_at.append(r_at["ttp"])
                    dose_mtd.append(r_mtd["cumulative_dose"]); dose_at.append(r_at["cumulative_dose"])
                except Exception:  # noqa: BLE001
                    continue
            if not ttp_mtd or not ttp_at:
                continue
            adv = float(np.median(ttp_at)) - float(np.median(ttp_mtd))
            mtd_frac = float(np.mean(dose_mtd) / np.mean(ttp_mtd)) if np.mean(ttp_mtd) > 0 else 0.0
            at_frac = float(np.mean(dose_at) / np.mean(ttp_at)) if np.mean(ttp_at) > 0 else 0.0
            ds = 1.0 - at_frac / max(mtd_frac, 1e-9)
            advantages.append(adv); drug_savings.append(ds)
        except Exception:  # noqa: BLE001
            continue
    if not advantages:
        return None
    advantages = np.array(advantages)
    drug_savings = np.array(drug_savings)
    return {
        "n_completed": len(advantages),
        "p_at50_wins_ttp": float(np.mean(advantages > 0)),
        "p_at50_wins_drug": float(np.mean(drug_savings > 0)),
        "median_advantage_d": float(np.median(advantages)),
        "median_drug_savings": float(np.median(drug_savings)),
    }


def main(seed: int = 0, n_posterior: int = 15, n_patients_per_draw: int = 2) -> None:
    warnings.filterwarnings("ignore")
    log.info(f"2D regime scan: K_TP_drop x alpha[T-,T+] (M6 follow-up)")

    K_TP_drop_grid = np.array([1000, 3000, 5000, 7000, 9000], dtype=float)
    alpha_grid = np.array([1.5, 2.5, 3.5, 5.0, 7.0], dtype=float)
    log.info(f"  Grid: {len(K_TP_drop_grid)} x {len(alpha_grid)} = {len(K_TP_drop_grid) * len(alpha_grid)} points")

    rng = np.random.default_rng(seed)
    P_grid = np.full((len(alpha_grid), len(K_TP_drop_grid)), np.nan)
    M_grid = np.full((len(alpha_grid), len(K_TP_drop_grid)), np.nan)
    D_grid = np.full((len(alpha_grid), len(K_TP_drop_grid)), np.nan)

    n_total = len(K_TP_drop_grid) * len(alpha_grid)
    progress = 0
    grid_results = []
    for i, alpha_val in enumerate(alpha_grid):
        for j, K_val in enumerate(K_TP_drop_grid):
            theta = _NOMINAL.copy()
            theta[3] = alpha_val  # alpha[T-,T+]
            theta[5] = K_val
            res = evaluate_grid_point(theta, n_posterior, n_patients_per_draw, rng)
            progress += 1
            if res is None:
                log.warning(f"  [{progress}/{n_total}] alpha={alpha_val}, K={K_val}: failed")
                continue
            P_grid[i, j] = res["p_at50_wins_ttp"]
            M_grid[i, j] = res["median_advantage_d"]
            D_grid[i, j] = res["median_drug_savings"]
            grid_results.append({
                "K_TP_drop": float(K_val), "alpha_T_minus_T_plus": float(alpha_val),
                **res,
            })
            log.info(
                f"  [{progress}/{n_total}] α={alpha_val}, K={K_val}: "
                f"P(AT50 wins TTP)={res['p_at50_wins_ttp']:.0%}, "
                f"median_adv={res['median_advantage_d']:.0f}d, "
                f"drug_savings={res['median_drug_savings']:.0%}"
            )

    # --- Figure ---
    fig = plt.figure(figsize=(15, 5.5))
    fig.set_facecolor("white")
    gs = fig.add_gridspec(1, 3, wspace=0.32)

    # Heatmap of P(AT50 wins TTP)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(P_grid, origin="lower", cmap="RdYlGn", vmin=0, vmax=1.0,
                      aspect="auto", interpolation="nearest")
    ax1.set_xticks(range(len(K_TP_drop_grid)))
    ax1.set_xticklabels([f"{K:.0f}" for K in K_TP_drop_grid])
    ax1.set_yticks(range(len(alpha_grid)))
    ax1.set_yticklabels([f"{a:.1f}" for a in alpha_grid])
    ax1.set_xlabel("K_TP_drop")
    ax1.set_ylabel("α(T-, T+)")
    ax1.set_title("P(AT50 wins TTP) over posterior", fontsize=10)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    for i in range(len(alpha_grid)):
        for j in range(len(K_TP_drop_grid)):
            v = P_grid[i, j]
            if np.isnan(v): continue
            ax1.text(j, i, f"{v:.0%}", ha="center", va="center",
                     color=("white" if v < 0.5 else "black"), fontsize=8)

    # Heatmap of median advantage in months
    ax2 = fig.add_subplot(gs[0, 1])
    M_months = M_grid / 30
    vmax = max(np.nanmax(np.abs(M_months)), 1)
    im2 = ax2.imshow(M_months, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      aspect="auto", interpolation="nearest")
    ax2.set_xticks(range(len(K_TP_drop_grid)))
    ax2.set_xticklabels([f"{K:.0f}" for K in K_TP_drop_grid])
    ax2.set_yticks(range(len(alpha_grid)))
    ax2.set_yticklabels([f"{a:.1f}" for a in alpha_grid])
    ax2.set_xlabel("K_TP_drop")
    ax2.set_ylabel("α(T-, T+)")
    ax2.set_title("Median AT50 advantage (months)", fontsize=10)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    for i in range(len(alpha_grid)):
        for j in range(len(K_TP_drop_grid)):
            v = M_months[i, j]
            if np.isnan(v): continue
            ax2.text(j, i, f"{v:+.1f}", ha="center", va="center",
                     color=("white" if abs(v) > 0.5 * vmax else "black"), fontsize=8)

    # Heatmap of drug savings
    ax3 = fig.add_subplot(gs[0, 2])
    D_pct = D_grid * 100
    im3 = ax3.imshow(D_pct, origin="lower", cmap="Purples", vmin=0, vmax=100,
                      aspect="auto", interpolation="nearest")
    ax3.set_xticks(range(len(K_TP_drop_grid)))
    ax3.set_xticklabels([f"{K:.0f}" for K in K_TP_drop_grid])
    ax3.set_yticks(range(len(alpha_grid)))
    ax3.set_yticklabels([f"{a:.1f}" for a in alpha_grid])
    ax3.set_xlabel("K_TP_drop")
    ax3.set_ylabel("α(T-, T+)")
    ax3.set_title("AT50 drug savings (% of MTD's drug)", fontsize=10)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    for i in range(len(alpha_grid)):
        for j in range(len(K_TP_drop_grid)):
            v = D_pct[i, j]
            if np.isnan(v): continue
            ax3.text(j, i, f"{v:.0f}%", ha="center", va="center",
                     color=("white" if v > 50 else "black"), fontsize=8)

    fig.suptitle(
        f"M6 follow-up — 2D regime scan of posterior-aware policy preference "
        f"(N_posterior={n_posterior} per cell, n_patients={n_patients_per_draw})",
        fontsize=12,
    )

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True,
        ).strip()
    except subprocess.CalledProcessError:
        sha = "uncommitted"
    date = dt.date.today().isoformat()
    base = _REPO_ROOT / "results" / "figures" / f"fig15_regime_scan_2d_{sha}_{date}"
    fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {base}.png and .pdf")

    summary = {
        "experiment": "regime_scan_2d",
        "git_sha": sha,
        "date": date,
        "scanned_axes": ["K_TP_drop", "alpha_T_minus_T_plus"],
        "K_TP_drop_grid": K_TP_drop_grid.tolist(),
        "alpha_grid": alpha_grid.tolist(),
        "n_posterior": int(n_posterior),
        "n_patients_per_draw": int(n_patients_per_draw),
        "P_AT50_wins_grid": P_grid.tolist(),
        "median_advantage_grid": M_grid.tolist(),
        "drug_savings_grid": D_grid.tolist(),
        "grid_results": grid_results,
        "headline": {
            "n_posterior_sensitive_cells": int(np.sum((P_grid > 0.10) & (P_grid < 0.90))),
            "n_at50_dominant_cells": int(np.sum(P_grid > 0.90)),
            "n_mtd_dominant_cells": int(np.sum(P_grid < 0.10)),
        },
    }
    summary_path = _REPO_ROOT / "results" / f"regime_scan_2d_summary_{sha}_{date}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D regime scan")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-posterior", type=int, default=15)
    parser.add_argument("--n-patients-per-draw", type=int, default=2)
    args = parser.parse_args()
    main(seed=args.seed, n_posterior=args.n_posterior, n_patients_per_draw=args.n_patients_per_draw)
