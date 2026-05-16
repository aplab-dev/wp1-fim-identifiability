"""Multi-modal observation channel FIM analysis — Phase 4 §4.2.

WP1 §4.3 concluded "Multi-modal observation channels (ctDNA, AR-V7 transcript,
mIHC tumor-infiltration data) are the only path to higher identifiability."
This experiment quantifies how much each candidate channel actually buys on
the 3-pop Zhang K-shift model.

Baseline (WP1 §3.2): under PSA-only observation at the canonical Zhang θ
with constant MTD, the 6-parameter FIM has effective rank 3 of 6 with
eigenvalues (3.7e8, 2.3e7, 3.6e5, 0.88, 2.8e-4, 8.7e-6). The three
unidentifiable directions are the rank-deficient gap that this experiment
tries to close.

Candidate observation channels (each modeled with realistic 10% relative noise):

  PSA       — aggregate cell-derived PSA filter (current baseline)
  TTB       — total tumor burden (T+ + TP + T-) from imaging
  T-_frac   — resistant fraction T-/total (e.g., from ctDNA with T- markers)
  TP        — TP cell count (e.g., from AR-V7 transcript, which is mostly
              produced by TP cells in the Zhang ontology)
  T+        — T+ cell count (e.g., from PSMA-PET, which targets T+ cells)

For each combination, we compute the FIM by stacking sensitivities from all
channels (which is mathematically equivalent to the sum of per-channel FIMs
when channels have independent noise), then report eigenvalue spectrum and
effective rank.

Output:
- ``results/figures/fig24_multimodal_fim_{git_sha}_{date}.{png,pdf}``
- ``results/multimodal_fim_summary_{git_sha}_{date}.json``

Headline metric per channel combination: effective rank, condition number,
smallest eigenvalue (= worst-case unidentifiable direction).
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
from simulators.lv_3pop_kshift import LV3PopKShift, LV3PopParams  # noqa: E402
from simulators.psa_dynamics import PSAParams, psa_steady_state  # noqa: E402
from zhang2017 import ZHANG_CANONICAL_X0, zhang_canonical_lv_params  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


PARAM_NAMES = ["r_T+", "r_TP", "r_T-", "α(T-,T+)", "α(T-,TP)", "K_TP_drop"]

_canon = zhang_canonical_lv_params()
PARAM_NOMINAL = np.array([
    _canon.r_Tplus, _canon.r_TP, _canon.r_Tminus,
    float(_canon.alpha[2, 0]), float(_canon.alpha[2, 1]),
    _canon.K_TP_drop,
])

X0 = ZHANG_CANONICAL_X0
PSA_PARAMS = PSAParams()
T_OBS = np.arange(0.0, 1500.0 + 1, 28.0)

# Relative observation noise — same 10% used in WP1 §3.2 for PSA.
NOISE_REL = 0.10
# Floor noise at this fraction of peak to avoid singular weighting.
NOISE_FLOOR_FRAC = 0.10


def _build_lv_params(theta: np.ndarray) -> LV3PopParams:
    r_Tplus, r_TP, r_Tminus, alpha_2_0, alpha_2_1, K_TP_drop = theta
    alpha = _canon.alpha.copy()
    alpha[2, 0] = alpha_2_0
    alpha[2, 1] = alpha_2_1
    return LV3PopParams(
        r_Tplus=r_Tplus, r_TP=r_TP, r_Tminus=r_Tminus,
        K_Tminus=_canon.K_Tminus, K_TP_max=_canon.K_TP_max,
        K_TP_drop=K_TP_drop, mu_max=_canon.mu_max, mu_drop=_canon.mu_drop,
        alpha=alpha,
    )


def simulate_trajectories(theta: np.ndarray) -> np.ndarray:
    """Simulate the 3-pop K-shift + PSA under MTD; return (T, 5) trajectory.

    Returns array with columns [T+, TP, T-, PSA, TTB].
    TTB = T+ + TP + T- (total tumor burden).
    """
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
    sol = None
    for method in ("LSODA", "BDF"):
        try:
            trial = solve_ivp(rhs, t_span=(0.0, T_OBS[-1]), y0=y0,
                              t_eval=T_OBS, method=method,
                              rtol=1e-8, atol=1e-3)
            if trial.success:
                sol = trial
                break
        except Exception:  # noqa: BLE001
            continue
    if sol is None or not sol.success:
        raise RuntimeError("3-pop multimodal predictor failed")
    # sol.y is (4, T). Add TTB.
    ttb = sol.y[:3].sum(axis=0)
    return np.column_stack([sol.y[0], sol.y[1], sol.y[2], sol.y[3], ttb])


# Channel extractors — given the (T, 5) trajectory matrix, return a (T,)
# observation per channel.
CHANNEL_EXTRACTORS = {
    "PSA":     lambda traj: traj[:, 3],
    "TTB":     lambda traj: traj[:, 4],
    "T-_frac": lambda traj: traj[:, 2] / np.maximum(traj[:, 4], 1.0),
    "TP":      lambda traj: traj[:, 1],
    "T+":      lambda traj: traj[:, 0],
}


def make_predict(channels: list[str]):
    """Build a `predict(theta) -> flattened multi-channel observation` closure.

    The concatenated trajectory has length len(channels) * T. Sigma is built
    matching this concatenation (per-channel rel-noise floor).
    """
    def predict(theta: np.ndarray) -> np.ndarray:
        traj = simulate_trajectories(theta)
        return np.concatenate([CHANNEL_EXTRACTORS[c](traj) for c in channels])
    return predict


def make_sigma(channels: list[str], theta: np.ndarray) -> np.ndarray:
    """Build the per-observation noise std array, matching predict output."""
    traj = simulate_trajectories(theta)
    parts = []
    for c in channels:
        y_c = CHANNEL_EXTRACTORS[c](traj)
        peak = max(float(np.max(np.abs(y_c))), 1e-6)
        sigma_c = NOISE_REL * np.maximum(np.abs(y_c), NOISE_FLOOR_FRAC * peak)
        parts.append(sigma_c)
    return np.concatenate(parts)


def effective_rank(eigvals: np.ndarray, ratio_threshold: float = 1e-6) -> int:
    """Number of eigenvalues above ratio_threshold * max(eigvals)."""
    eigval_max = eigvals.max()
    return int(np.sum(eigvals > ratio_threshold * eigval_max))


def analyze_channel_combo(channels: list[str], theta_nominal: np.ndarray) -> dict:
    """Compute FIM, eigenvalues, rank for a given channel combination."""
    predict = make_predict(channels)
    sigma = make_sigma(channels, theta_nominal)
    fim_result = compute_fim(
        predict, theta_nominal,
        eps_rel=1e-3, sigma=sigma, param_names=PARAM_NAMES,
    )
    eigvals = np.linalg.eigvalsh(fim_result.fim)
    eigvals = np.sort(eigvals)[::-1]  # descending
    return {
        "channels": channels,
        "n_channels": len(channels),
        "eigenvalues": eigvals.tolist(),
        "eigvalue_max": float(eigvals.max()),
        "eigvalue_min": float(eigvals.min()),
        "condition_number": float(eigvals.max() / max(eigvals.min(), 1e-300)),
        "effective_rank_1e_minus_6": effective_rank(eigvals, 1e-6),
        "effective_rank_1e_minus_3": effective_rank(eigvals, 1e-3),
    }


def main(seed: int = 0) -> None:
    warnings.filterwarnings("ignore")
    log.info("Multi-modal observation channel FIM analysis (Phase 4 §4.2)")
    log.info(f"Nominal θ: {PARAM_NOMINAL.tolist()}")
    log.info(f"Schedule: constant MTD, {len(T_OBS)} obs at 28-day cadence over 1500 days")

    # The channel combos to test, ordered by clinical plausibility
    combos = [
        ["PSA"],                                # Baseline (WP1 §3.2)
        ["PSA", "TTB"],                          # PSA + imaging
        ["PSA", "T-_frac"],                      # PSA + ctDNA-derived resistant fraction
        ["PSA", "TP"],                           # PSA + AR-V7 transcript
        ["PSA", "T+"],                           # PSA + PSMA-PET
        ["PSA", "TTB", "T-_frac"],               # PSA + imaging + ctDNA
        ["PSA", "TTB", "TP"],                    # PSA + imaging + AR-V7
        ["PSA", "TTB", "T-_frac", "TP"],         # Comprehensive (4 channels)
        ["PSA", "TTB", "T-_frac", "TP", "T+"],   # Full 5-channel (idealized)
    ]

    results = []
    for combo in combos:
        log.info(f"  Channel combo: {combo}")
        try:
            r = analyze_channel_combo(combo, PARAM_NOMINAL)
            results.append(r)
            log.info(
                f"    eigenvalues: {['%.2e' % e for e in r['eigenvalues']]}"
            )
            log.info(
                f"    rank (1e-6 / 1e-3): {r['effective_rank_1e_minus_6']} / "
                f"{r['effective_rank_1e_minus_3']}, "
                f"κ = {r['condition_number']:.2e}"
            )
        except Exception as e:  # noqa: BLE001
            log.warning(f"    failed: {e}")
            continue

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor="white")
    n_combos = len(results)
    x = np.arange(n_combos)
    combo_labels = ["+".join(r["channels"]) for r in results]

    # Panel A: eigenvalue spectrum per combo
    ax_a = axes[0]
    for k in range(6):
        ax_a.semilogy(
            x, [r["eigenvalues"][k] for r in results],
            "o-", label=f"λ_{k+1}", alpha=0.85,
        )
    ax_a.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="threshold λ=1")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(combo_labels, rotation=30, ha="right", fontsize=7)
    ax_a.set_ylabel("Eigenvalue (log scale)")
    ax_a.set_title("[A] FIM eigenvalue spectrum vs observation channel combo", fontsize=10)
    ax_a.legend(fontsize=7, ncol=2, loc="lower right")
    ax_a.grid(True, alpha=0.3, which="both")

    # Panel B: effective rank
    ax_b = axes[1]
    ranks_1e6 = [r["effective_rank_1e_minus_6"] for r in results]
    ranks_1e3 = [r["effective_rank_1e_minus_3"] for r in results]
    ax_b.bar(x - 0.2, ranks_1e6, width=0.4, color="tab:blue",
             alpha=0.7, label="Rank (ratio > 1e-6)")
    ax_b.bar(x + 0.2, ranks_1e3, width=0.4, color="tab:red",
             alpha=0.7, label="Rank (ratio > 1e-3, stricter)")
    ax_b.axhline(6.0, color="gray", linestyle="--", alpha=0.5, label="full rank = 6")
    ax_b.axhline(3.0, color="tab:gray", linestyle=":", alpha=0.5, label="PSA-only rank = 3")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(combo_labels, rotation=30, ha="right", fontsize=7)
    ax_b.set_ylim(0, 6.5)
    ax_b.set_ylabel("Effective rank")
    ax_b.set_title("[B] Effective FIM rank vs channel combo", fontsize=10)
    ax_b.legend(fontsize=8, loc="upper left")
    ax_b.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Multi-modal observation channel FIM rank (3-pop Zhang K-shift, canonical θ, MTD schedule)",
        fontsize=12,
    )
    fig.tight_layout()

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True,
        ).strip()
    except subprocess.CalledProcessError:
        sha = "uncommitted"
    date = dt.date.today().isoformat()
    base = _REPO_ROOT / "results" / "figures" / f"fig24_multimodal_fim_{sha}_{date}"
    fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {base}.png and .pdf")

    summary = {
        "experiment": "multimodal_fim",
        "git_sha": sha,
        "date": date,
        "nominal_theta": PARAM_NOMINAL.tolist(),
        "param_names": PARAM_NAMES,
        "schedule": "constant MTD, 28-day cadence, 1500 days",
        "noise_model": f"{int(NOISE_REL*100)}% relative + floor {int(NOISE_FLOOR_FRAC*100)}% of peak",
        "headline": {
            "psa_only_rank": results[0]["effective_rank_1e_minus_6"] if results else None,
            "full_rank_target": 6,
            "best_combo": None,
            "best_combo_rank": None,
        },
        "results_per_combo": results,
    }
    # Find the simplest combo achieving full rank
    for r in results:
        if r["effective_rank_1e_minus_6"] >= 6:
            summary["headline"]["best_combo"] = r["channels"]
            summary["headline"]["best_combo_rank"] = 6
            break
    summary_path = _REPO_ROOT / "results" / f"multimodal_fim_summary_{sha}_{date}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-modal observation channel FIM analysis")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(seed=args.seed)
