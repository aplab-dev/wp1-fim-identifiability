"""Regime scan — find clinical regimes where policy preference is posterior-sensitive.

Phase 3 §3.2 deliverable. Experiment 09 showed P(AT50 wins TTP) = 100% on
the canonical Zhang regime. Experiment 10 confirmed this on the actual MCMC
posterior. Both findings are for ONE regime.

Question: do other clinical regimes give posterior-SENSITIVE policy
preferences (where P(AT50 wins TTP) < 100%)? Such regimes are where
posterior-aware control matters more than point-estimate optimal control —
the central methodological claim of Candidate C.

Approach: scan along a parameter axis that shifts the AT50-vs-MTD
tradeoff. Two natural candidates:
- ``K_TP_drop`` — drug effectiveness on TP carrying capacity. Lower
  K_TP_drop = drug less effective. As K_TP_drop → 0, MTD becomes weaker
  → AT50 advantage shrinks → may become posterior-sensitive.
- ``alpha_T-, T+`` — strength of T- suppression by T+. As this drops
  toward 1.0, T- is less suppressed at no-drug equilibrium → resistance
  reservoir grows → AT50 cycling becomes less effective.

Per scan point:
1. Set theta_0 with the scanned parameter at its scan value.
2. Compute FIM at theta_0.
3. Sample N=30 posterior draws from FIM-induced regularized Gaussian.
4. For each accepted draw: run cohort (n=3 patients) under MTD and AT50.
5. Compute P(AT50 wins TTP), median advantage, drug savings.

Output:
- ``results/figures/fig13_regime_scan_{git_sha}_{date}.{png,pdf}``
  showing P(AT50 wins) and median advantage as a function of the scanned
  parameter. Identifies the regime boundary where the policy comparison
  becomes uncertain.
- ``results/regime_scan_summary_{git_sha}_{date}.json``.
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
    r_Tplus, r_TP, r_Tminus, alpha_2_0, alpha_2_1, K_TP_drop = theta
    if r_Tplus <= 0 or r_TP <= 0 or r_Tminus <= 0: return False
    if alpha_2_0 < 0 or alpha_2_1 < 0: return False
    if K_TP_drop < 0 or K_TP_drop > _canon.K_TP_max: return False
    return True


def evaluate_regime(theta_center: np.ndarray, n_posterior: int, n_patients: int,
                    rng: np.random.Generator) -> dict | None:
    """Return P(AT50 wins TTP), median advantage, drug savings for one regime."""
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
            lv = _build_lv_params(draw)
            params_for_runner = ZhangPatientParams(lv_params=lv)
            ttp_mtd, ttp_at50 = [], []
            dose_mtd, dose_at50 = [], []
            ss = np.random.SeedSequence(int(rng.integers(0, 1_000_000)))
            children = ss.spawn(n_patients)
            for child in children:
                patient_rng = np.random.Generator(np.random.PCG64(child))
                try:
                    r_mtd = run_zhang_patient(params_for_runner, MTDPolicy(), rng=patient_rng)
                    r_at = run_zhang_patient(params_for_runner, AT50Policy(), rng=patient_rng)
                    ttp_mtd.append(r_mtd["ttp"])
                    ttp_at50.append(r_at["ttp"])
                    dose_mtd.append(r_mtd["cumulative_dose"])
                    dose_at50.append(r_at["cumulative_dose"])
                except Exception:  # noqa: BLE001
                    continue
            if not ttp_mtd or not ttp_at50:
                continue
            adv = float(np.median(ttp_at50)) - float(np.median(ttp_mtd))
            mtd_frac = float(np.mean(dose_mtd) / np.mean(ttp_mtd)) if np.mean(ttp_mtd) > 0 else 0.0
            at_frac = float(np.mean(dose_at50) / np.mean(ttp_at50)) if np.mean(ttp_at50) > 0 else 0.0
            ds = 1.0 - at_frac / max(mtd_frac, 1e-9)
            advantages.append(adv)
            drug_savings.append(ds)
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
        "iqr_advantage": [float(np.percentile(advantages, 25)),
                          float(np.percentile(advantages, 75))],
        "median_drug_savings": float(np.median(drug_savings)),
    }


def main(seed: int = 0, n_posterior: int = 30, n_patients_per_draw: int = 3) -> None:
    warnings.filterwarnings("ignore")

    log.info(f"Regime scan: K_TP_drop axis (Phase 3 §3.2)")
    log.info(f"  Settings: N_posterior={n_posterior}, n_patients_per_draw={n_patients_per_draw}")

    # Scan K_TP_drop from "weak drug" (1000) to "strong drug" (9900, canonical)
    K_TP_drop_grid = np.array([1000, 2500, 4000, 5500, 7000, 8500, 9900], dtype=float)
    rng = np.random.default_rng(seed)

    results = []
    for K in K_TP_drop_grid:
        theta = _NOMINAL.copy()
        theta[5] = K  # K_TP_drop is index 5
        log.info(f"  K_TP_drop = {K}: evaluating regime...")
        regime_result = evaluate_regime(theta, n_posterior, n_patients_per_draw, rng)
        if regime_result is None:
            log.warning(f"    failed to evaluate regime at K_TP_drop = {K}")
            continue
        regime_result["K_TP_drop"] = float(K)
        results.append(regime_result)
        log.info(
            f"    P(AT50 wins TTP) = {regime_result['p_at50_wins_ttp']:.0%}, "
            f"median adv = {regime_result['median_advantage_d']:.0f} d, "
            f"P(AT50 wins drug) = {regime_result['p_at50_wins_drug']:.0%}, "
            f"completed = {regime_result['n_completed']}/{n_posterior}"
        )

    if not results:
        log.error("No regime evaluations succeeded — bailing out")
        return

    K_grid = np.array([r["K_TP_drop"] for r in results])
    p_ttp = np.array([r["p_at50_wins_ttp"] for r in results])
    p_drug = np.array([r["p_at50_wins_drug"] for r in results])
    medians = np.array([r["median_advantage_d"] for r in results])
    iqr_low = np.array([r["iqr_advantage"][0] for r in results])
    iqr_high = np.array([r["iqr_advantage"][1] for r in results])
    drug_savings = np.array([r["median_drug_savings"] for r in results])

    # --- Figure ---
    fig = plt.figure(figsize=(13, 9))
    fig.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.32)

    # P(AT50 wins) vs K_TP_drop
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(K_grid, p_ttp, color="tab:red", linewidth=2.0, marker="o", label="P(AT50 wins TTP)")
    ax1.plot(K_grid, p_drug, color="tab:purple", linewidth=2.0, marker="s", linestyle="--", label="P(AT50 wins drug)")
    ax1.axhline(0.5, color="tab:gray", linestyle=":", linewidth=1.0, alpha=0.6, label="coin-flip")
    ax1.axhline(1.0, color="tab:gray", linestyle=":", linewidth=0.8, alpha=0.4)
    ax1.set_xlabel("K_TP_drop (drug effectiveness on TP carrying capacity)")
    ax1.set_ylabel("P(AT50 wins) across posterior")
    ax1.set_title("Posterior-aware policy preference vs drug effectiveness", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.02, 1.05)

    # Median advantage vs K_TP_drop with IQR shading
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(K_grid, iqr_low / 30, iqr_high / 30, color="tab:red", alpha=0.2, label="IQR")
    ax2.plot(K_grid, medians / 30, color="tab:red", linewidth=2.0, marker="o", label="median")
    ax2.axhline(0, color="black", linewidth=0.7)
    ax2.set_xlabel("K_TP_drop")
    ax2.set_ylabel("AT50 TTP advantage (months)")
    ax2.set_title("AT50 TTP advantage vs drug effectiveness", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Drug savings vs K_TP_drop
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(K_grid, drug_savings * 100, color="tab:purple", linewidth=2.0, marker="^")
    ax3.axhline(0, color="black", linewidth=0.7)
    ax3.set_xlabel("K_TP_drop")
    ax3.set_ylabel("AT50 drug savings (% of MTD's drug)")
    ax3.set_title("AT50 drug-burden reduction vs drug effectiveness", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Summary text panel
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    sensitive_idx = np.where(p_ttp < 0.99)[0]
    if len(sensitive_idx) > 0:
        sensitive_K = K_grid[sensitive_idx]
        sensitive_msg = (
            f"Posterior-SENSITIVE regimes found:\n"
            f"  K_TP_drop ∈ {sensitive_K.tolist()}\n"
            f"  P(AT50 wins TTP) drops to {p_ttp[sensitive_idx].min():.0%}\n\n"
            f"This is where posterior-aware control matters\n"
            f"more than point-estimate control.\n"
        )
    else:
        sensitive_msg = (
            f"No posterior-sensitive regimes found in the\n"
            f"K_TP_drop ∈ [{K_grid.min():.0f}, {K_grid.max():.0f}] range.\n\n"
            f"P(AT50 wins TTP) stays at 100% across the scan.\n"
            f"In this scanned axis, AT50 robustly dominates.\n"
        )
    summary_text = (
        f"REGIME SCAN RESULTS\n\n"
        f"Scanned: K_TP_drop on canonical Zhang θ\n"
        f"Range: [{K_grid.min():.0f}, {K_grid.max():.0f}]\n"
        f"N_posterior: {n_posterior}\n"
        f"n_patients_per_draw: {n_patients_per_draw}\n"
        f"Total simulations: ~{n_posterior * n_patients_per_draw * 2 * len(K_grid)}\n\n"
        f"{sensitive_msg}\n"
        f"This experiment: WP1 §3.6 candidate result.\n"
        f"Phase 3 §3.2 main deliverable."
    )
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    fig.suptitle(
        "Phase 3 §3.2 regime scan — does the AT50 vs MTD posterior preference depend on drug effectiveness?",
        fontsize=11,
    )

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True,
        ).strip()
    except subprocess.CalledProcessError:
        sha = "uncommitted"
    date = dt.date.today().isoformat()
    base = _REPO_ROOT / "results" / "figures" / f"fig13_regime_scan_{sha}_{date}"
    fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {base}.png and .pdf")

    summary = {
        "experiment": "regime_scan_K_TP_drop",
        "git_sha": sha,
        "date": date,
        "scanned_axis": "K_TP_drop",
        "K_TP_drop_grid": K_grid.tolist(),
        "n_posterior": int(n_posterior),
        "n_patients_per_draw": int(n_patients_per_draw),
        "results": results,
        "headline": {
            "n_regimes_scanned": len(results),
            "p_at50_wins_ttp_min": float(p_ttp.min()),
            "p_at50_wins_ttp_max": float(p_ttp.max()),
            "any_posterior_sensitive": bool(np.any(p_ttp < 0.99)),
            "min_K_TP_drop_with_full_robustness": (
                float(K_grid[p_ttp >= 0.99].min()) if np.any(p_ttp >= 0.99) else None
            ),
        },
    }
    summary_path = _REPO_ROOT / "results" / f"regime_scan_summary_{sha}_{date}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regime scan for posterior-aware policy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-posterior", type=int, default=30)
    parser.add_argument("--n-patients-per-draw", type=int, default=3)
    args = parser.parse_args()
    main(seed=args.seed, n_posterior=args.n_posterior, n_patients_per_draw=args.n_patients_per_draw)
