"""FIM identifiability — 3-pop K-shift Zhang model under MTD.

Extension of Stage 2.5b's FIM analysis from the 2-pop multdeath model to
the clinical-tribe 3-pop K-shift model used by Zhang 2017 / Cunningham 2020.
Phase 3 Candidate C (Online Bayesian ID + control on the clinical model)
needs the 3-pop FIM as its structural foundation.

Parameter space (6 parameters, biologically meaningful subset):
- r_Tplus, r_TP, r_Tminus — growth rates (per day)
- alpha[2, 0] — T- suppression by T+ (cross-pop competition)
- alpha[2, 1] — T- suppression by TP
- K_TP_drop — drug-induced collapse of TP carrying capacity

The other parameters (K_Tminus, K_TP_max, mu_max, mu_drop, alpha[0,*],
alpha[1,*]) are held fixed at their canonical Zhang values. This 6-D
subset is what a clinical fit would actually try to identify per patient
(growth rates + competition + drug effectiveness).

Observation: PSA at 28-day clinical-lab cadence over 1500 days, 10%
relative noise. Schedule: constant MTD (most informative single schedule
per Stage 2.5b finding).

Hypothesis (from Stage 2.5b experience): rank will be > 1 because
3-pop dynamics has more time-scale structure (T+ collapses fast, T-
grows slow, TP is intermediate). Each component of the trajectory
gives information about a different parameter combination.

Output:
- ``results/figures/fig08_fim_3pop_zhang_{git_sha}_{date}.{png,pdf}``
- ``results/fim_3pop_summary_{git_sha}_{date}.json``
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

from identifiability import compute_fim, fim_eigendecomposition  # noqa: E402
from simulators.lv_3pop_kshift import LV3PopKShift, LV3PopParams  # noqa: E402
from simulators.psa_dynamics import PSAParams, psa_steady_state  # noqa: E402
from zhang2017 import ZHANG_CANONICAL_X0, zhang_canonical_lv_params  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


PARAM_NAMES = ["r_T+", "r_TP", "r_T-", "α(T-,T+)", "α(T-,TP)", "K_TP_drop"]

# Nominal vector — extracted from Zhang canonical params
_canon = zhang_canonical_lv_params()
PARAM_NOMINAL = np.array([
    _canon.r_Tplus,        # 2.7726e-3
    _canon.r_TP,           # 3.4657e-3
    _canon.r_Tminus,       # 6.6542e-3
    float(_canon.alpha[2, 0]),  # 3.0
    float(_canon.alpha[2, 1]),  # 4.0
    _canon.K_TP_drop,      # 9_900
])

# Fixed parameters (held at canonical Zhang values)
X0 = ZHANG_CANONICAL_X0
PSA_PARAMS = PSAParams()  # phi=0.5, rho=1.0
T_OBS = np.arange(0.0, 1500.0 + 1, 28.0)  # 4-week labs over 50 months


def _build_lv_params(theta: np.ndarray) -> LV3PopParams:
    """Build a fresh LV3PopParams from a 6-parameter vector."""
    r_Tplus, r_TP, r_Tminus, alpha_2_0, alpha_2_1, K_TP_drop = theta
    # Build alpha matrix: keep canonical except for the two T- suppression entries
    alpha = _canon.alpha.copy()
    alpha[2, 0] = alpha_2_0
    alpha[2, 1] = alpha_2_1
    return LV3PopParams(
        r_Tplus=r_Tplus,
        r_TP=r_TP,
        r_Tminus=r_Tminus,
        K_Tminus=_canon.K_Tminus,
        K_TP_max=_canon.K_TP_max,
        K_TP_drop=K_TP_drop,
        mu_max=_canon.mu_max,
        mu_drop=_canon.mu_drop,
        alpha=alpha,
    )


def predict_psa_under_mtd(theta: np.ndarray) -> np.ndarray:
    """Simulate the 3-pop K-shift + PSA filter under constant MTD; return PSA at T_OBS."""
    lv = _build_lv_params(theta)
    sim = LV3PopKShift(lv)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        x = y[:3]
        psa = y[3]
        dx = sim.dynamics(t, x, Lambda=1.0)
        dpsa = PSA_PARAMS.rho * float(np.sum(x)) - PSA_PARAMS.phi * psa
        return np.concatenate([dx, [dpsa]])

    psa0 = psa_steady_state(float(np.sum(X0)), PSA_PARAMS)
    y0 = np.array([X0[0], X0[1], X0[2], psa0])

    # LSODA -> BDF fallback (stiff under MTD)
    sol = None
    last_msg = ""
    for method in ("LSODA", "BDF"):
        try:
            trial = solve_ivp(
                rhs,
                t_span=(0.0, T_OBS[-1]),
                y0=y0,
                t_eval=T_OBS,
                method=method,
                rtol=1e-8,
                atol=1e-3,
            )
            if trial.success:
                sol = trial
                break
            last_msg = trial.message
        except Exception as e:  # noqa: BLE001
            last_msg = str(e)
    if sol is None or not sol.success:
        raise RuntimeError(f"3-pop FIM predictor failed (LSODA + BDF): {last_msg}")
    return sol.y[3]


def main(seed: int = 0) -> None:
    warnings.filterwarnings("ignore")
    log.info("FIM identifiability on 3-pop K-shift Zhang model under MTD")
    log.info(f"  6 parameters: {PARAM_NAMES}")
    log.info(f"  Nominal: {PARAM_NOMINAL.tolist()}")
    log.info(f"  Observation: {len(T_OBS)} PSA points at 28-day cadence over 1500 days")

    psa_nom = predict_psa_under_mtd(PARAM_NOMINAL)
    log.info(
        f"  Nominal PSA: min={psa_nom.min():.0f}, max={psa_nom.max():.0f}, "
        f"baseline={psa_nom[0]:.0f}, final={psa_nom[-1]:.0f}"
    )

    sigma = 0.10 * np.maximum(psa_nom, 0.1 * psa_nom.max())

    fim_result = compute_fim(
        predict=predict_psa_under_mtd,
        theta_nominal=PARAM_NOMINAL,
        eps_rel=1e-3,
        sigma=sigma,
        param_names=PARAM_NAMES,
    )
    decomp = fim_eigendecomposition(fim_result)

    log.info(f"  Effective rank: {decomp['effective_rank']} / {decomp['n_params']}")
    log.info(f"  Eigenvalues: {decomp['eigenvalues'].tolist()}")
    log.info(f"  Condition number: {decomp['condition_number']:.2e}")

    # --- Figure ---
    fig = plt.figure(figsize=(14, 11))
    fig.set_facecolor("white")
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1], hspace=0.45, wspace=0.32)

    # Panel 1: nominal PSA trajectory
    ax_psa = fig.add_subplot(gs[0, 0])
    ax_psa.plot(T_OBS, psa_nom, color="tab:purple", linewidth=1.6, marker="o", markersize=3, label="PSA(t)")
    ax_psa.fill_between(T_OBS, psa_nom - sigma, psa_nom + sigma, color="tab:purple", alpha=0.15, label="±σ (10% rel.)")
    ax_psa.set_xlabel("Time (days)")
    ax_psa.set_ylabel("PSA")
    ax_psa.set_title("Nominal PSA trajectory under MTD\n(3-pop K-shift, Zhang canonical params)", fontsize=10)
    ax_psa.legend(fontsize=9)
    ax_psa.grid(True, alpha=0.3)

    # Panel 2: FIM heatmap (log-scale)
    ax_fim = fig.add_subplot(gs[0, 1])
    fim_log = np.log10(np.abs(fim_result.fim) + 1e-30)
    n = len(PARAM_NAMES)
    im = ax_fim.imshow(fim_log, cmap="viridis", aspect="auto")
    ax_fim.set_xticks(range(n))
    ax_fim.set_yticks(range(n))
    ax_fim.set_xticklabels(PARAM_NAMES, rotation=45, ha="right")
    ax_fim.set_yticklabels(PARAM_NAMES)
    ax_fim.set_title("FIM (log₁₀|entries|)", fontsize=11)
    fig.colorbar(im, ax=ax_fim, fraction=0.046, pad=0.04)
    for i in range(n):
        for j in range(n):
            ax_fim.text(j, i, f"{fim_log[i, j]:.1f}", ha="center", va="center", color="white", fontsize=7)

    # Panel 3: eigenvalue spectrum
    ax_spec = fig.add_subplot(gs[1, 0])
    eigs = decomp["eigenvalues"]
    indices = np.arange(len(eigs))
    bar_colors = ["tab:green" if e > 1e-6 * eigs[0] else "tab:red" for e in eigs]
    ax_spec.bar(indices, eigs, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax_spec.set_yscale("log")
    ax_spec.set_xticks(indices)
    ax_spec.set_xticklabels([f"λ_{i+1}" for i in indices])
    ax_spec.set_title(
        f"FIM eigenvalue spectrum (effective rank = {decomp['effective_rank']}/{decomp['n_params']})",
        fontsize=11,
    )
    ax_spec.set_ylabel("Eigenvalue (log scale)")
    ax_spec.axhline(1e-6 * eigs[0], color="tab:red", linestyle="--", linewidth=1.0, alpha=0.5,
                    label="threshold (10⁻⁶ × λ_max)")
    ax_spec.grid(True, alpha=0.3, axis="y", which="both")
    ax_spec.legend(fontsize=9)

    # Panel 4: most + least identifiable directions
    ax_dir = fig.add_subplot(gs[1, 1])
    width = 0.4
    x = np.arange(n)
    most = decomp["most_identifiable_direction"]
    least = decomp["least_identifiable_direction"]
    if abs(most[np.argmax(np.abs(most))]) > 0:
        most = most * np.sign(most[np.argmax(np.abs(most))])
    if abs(least[np.argmax(np.abs(least))]) > 0:
        least = least * np.sign(least[np.argmax(np.abs(least))])
    ax_dir.bar(x - width / 2, most, width, color="tab:green", label=f"most identifiable (λ_1={eigs[0]:.2g})")
    ax_dir.bar(x + width / 2, least, width, color="tab:red", label=f"least identifiable (λ_{n}={eigs[-1]:.2g})")
    ax_dir.axhline(0, color="black", linewidth=0.5)
    ax_dir.set_xticks(x)
    ax_dir.set_xticklabels(PARAM_NAMES, rotation=20, ha="right")
    ax_dir.set_ylabel("Eigenvector component")
    ax_dir.set_title("Identifiability eigenvectors", fontsize=11)
    ax_dir.legend(fontsize=8)
    ax_dir.grid(True, alpha=0.3, axis="y")

    # Panel 5: estimate-correlation matrix
    ax_corr = fig.add_subplot(gs[2, 0])
    fim_inv = np.linalg.pinv(fim_result.fim)
    diag_sqrt = np.sqrt(np.maximum(np.diag(fim_inv), 0))
    denom = np.outer(diag_sqrt, diag_sqrt) + 1e-30
    corr = fim_inv / denom
    im2 = ax_corr.imshow(corr, cmap="RdBu_r", vmin=-1.05, vmax=1.05, aspect="auto")
    ax_corr.set_xticks(range(n))
    ax_corr.set_yticks(range(n))
    ax_corr.set_xticklabels(PARAM_NAMES, rotation=45, ha="right")
    ax_corr.set_yticklabels(PARAM_NAMES)
    ax_corr.set_title("Estimate-correlation matrix (FIM⁻¹ normalized)", fontsize=11)
    fig.colorbar(im2, ax=ax_corr, fraction=0.046, pad=0.04)
    for i in range(n):
        for j in range(n):
            ax_corr.text(j, i, f"{corr[i, j]:.2f}",
                         ha="center", va="center",
                         color=("white" if abs(corr[i, j]) > 0.5 else "black"), fontsize=7)

    # Panel 6: per-parameter sensitivity time series
    ax_sens = fig.add_subplot(gs[2, 1])
    for i, name in enumerate(PARAM_NAMES):
        s = fim_result.sensitivities[i]
        norm = np.max(np.abs(s)) + 1e-30
        ax_sens.plot(T_OBS, s / norm, linewidth=1.4, label=f"∂PSA/∂{name}")
    ax_sens.set_xlabel("Time (days)")
    ax_sens.set_ylabel("Normalized sensitivity")
    ax_sens.set_title("Per-parameter sensitivity over time", fontsize=11)
    ax_sens.legend(fontsize=7, loc="best")
    ax_sens.grid(True, alpha=0.3)
    ax_sens.axhline(0, color="black", linewidth=0.5)

    fig.suptitle(
        f"FIM identifiability — 3-pop K-shift Zhang model under MTD, PSA-only obs (N={len(T_OBS)} samples)",
        fontsize=13,
    )

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True,
        ).strip()
    except subprocess.CalledProcessError:
        sha = "uncommitted"
    date = dt.date.today().isoformat()
    base = _REPO_ROOT / "results" / "figures" / f"fig08_fim_3pop_zhang_{sha}_{date}"
    fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {base}.png and .pdf")

    # Comparison to Stage 2.5b
    summary = {
        "experiment": "fim_3pop_zhang_under_mtd",
        "git_sha": sha,
        "date": date,
        "model": "3-pop K-shift (Zhang canonical params)",
        "param_names": PARAM_NAMES,
        "param_nominal": PARAM_NOMINAL.tolist(),
        "fixed_params": {
            "K_Tminus": _canon.K_Tminus,
            "K_TP_max": _canon.K_TP_max,
            "mu_max": _canon.mu_max,
            "mu_drop": _canon.mu_drop,
            "alpha_off_diagonal_T_plus_TP_block": [
                [float(_canon.alpha[0, 1]), float(_canon.alpha[0, 2])],
                [float(_canon.alpha[1, 0]), float(_canon.alpha[1, 2])],
            ],
            "x0": list(X0),
            "psa_params": {"phi": PSA_PARAMS.phi, "rho": PSA_PARAMS.rho},
        },
        "observation": {
            "schedule": "MTD only",
            "n_obs": len(T_OBS),
            "t_max_days": float(T_OBS[-1]),
            "noise_model": "10% relative, floored at 10% of peak PSA",
        },
        "fim_eigenvalues": decomp["eigenvalues"].tolist(),
        "effective_rank": int(decomp["effective_rank"]),
        "n_params": int(decomp["n_params"]),
        "rank_deficient": bool(decomp["rank_deficient"]),
        "condition_number": float(decomp["condition_number"]),
        "most_identifiable_direction": [float(v) for v in most],
        "least_identifiable_direction": [float(v) for v in least],
        "comparison_to_stage_2_5b": {
            "stage_2_5b_2pop_rank": "1 / 4",
            "this_3pop_rank": f"{decomp['effective_rank']} / {decomp['n_params']}",
            "interpretation": (
                "3-pop has more time-scale structure (fast T+/TP collapse vs slow T- regrowth), "
                "which the FIM can resolve. If rank > 1, schedule cycling could expose still more "
                "directions; if rank still 1, the rank deficiency is even more severe than 2-pop."
            ),
        },
    }
    summary_path = (
        _REPO_ROOT / "results" / f"fim_3pop_summary_{sha}_{date}.json"
    )
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-pop K-shift FIM analysis")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (unused for deterministic FIM)")
    args = parser.parse_args()
    main(seed=args.seed)
