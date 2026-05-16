"""FIM identifiability analysis on 2-pop L-V multdeath model.

Applies the Fisher Information Matrix machinery to the 2-pop multiplicative-
death model under MTD treatment. Computes sensitivities of the PSA trajectory
to each parameter and decomposes the FIM to assess identifiability.

The model has 6 parameters:
- r_S, r_R (growth rates)
- alpha, beta (competition coefficients)
- K (carrying capacity — typically fixed at 1.0 for nondimensionalization)
- d (drug-induced death rate)

Plus the PSA filter has rho (production) and phi (decay), and there's
the IC (S0, R0). For this analysis we focus on the 4 *dynamics* parameters
(r_S, r_R, alpha, beta) treating K, d, ρ, φ, S0, R0 as fixed.

**Hypothesis from Phase 1 derivation 2:** The 4-parameter dynamics are
*not* fully identifiable from PSA-only observation. The expected effective
rank is ≤ 3.

The MTD trajectory is the most informative single schedule because:
- During the drug-on phase, S decays and R grows — sensitivities to all
  four parameters become time-resolved.
- A no-drug trajectory only gives information about R/S ratio at
  equilibrium, not separable parameters.

Output:
- ``results/figures/fig04_fim_identifiability_{git_sha}_{date}.{png,pdf}``
  with: FIM heatmap, eigenvalue spectrum, dominant eigenvector bar charts,
  parameter-pair correlation map.
- ``results/fim_summary_{git_sha}_{date}.json`` with effective rank,
  condition number, identifiable-direction analysis.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from identifiability import compute_fim, fim_eigendecomposition  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


PARAM_NAMES = ["r_S", "r_R", "alpha", "beta"]
PARAM_NOMINAL = np.array([0.05, 0.04, 0.7, 0.6])
# Fixed (non-identified) parameters
K_FIXED = 1.0
D_FIXED = 1.5
RHO_FIXED = 1.0
PHI_FIXED = 0.5
S0_FIXED = 0.6
R0_FIXED = 0.006

# Observation schedule: 4-week labs over a 500-day MTD trajectory.
T_OBS = np.arange(0.0, 500.0 + 1, 28.0)


def predict_psa(theta: np.ndarray) -> np.ndarray:
    """Simulate the 2-pop multdeath under MTD and return PSA at T_OBS.

    State vector: [S, R, PSA]. Coupled integration.
    """
    r_S, r_R, alpha, beta = theta

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        S, R, PSA = y
        dS = r_S * S * (1 - (S + alpha * R) / K_FIXED) - D_FIXED * 1.0 * S
        dR = r_R * R * (1 - (R + beta * S) / K_FIXED)
        dPSA = RHO_FIXED * (S + R) - PHI_FIXED * PSA
        return np.array([dS, dR, dPSA])

    psa_baseline = RHO_FIXED * (S0_FIXED + R0_FIXED) / PHI_FIXED
    sol = solve_ivp(
        rhs,
        t_span=(0.0, T_OBS[-1]),
        y0=np.array([S0_FIXED, R0_FIXED, psa_baseline]),
        t_eval=T_OBS,
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"FIM predictor solve_ivp failed: {sol.message}")
    return sol.y[2]  # PSA trajectory at T_OBS


def main(seed: int = 0) -> None:
    log.info(f"FIM identifiability analysis on 2-pop multdeath under MTD (seed={seed})")
    log.info(f"  Parameters: {PARAM_NAMES} = {PARAM_NOMINAL.tolist()}")
    log.info(f"  Observation: PSA at {len(T_OBS)} time points (28-day cadence, 500-day horizon)")

    # Simulate the nominal trajectory for the figure.
    psa_nom = predict_psa(PARAM_NOMINAL)
    log.info(
        f"  Nominal PSA trajectory: min={psa_nom.min():.3f}, max={psa_nom.max():.3f}, "
        f"baseline={psa_nom[0]:.3f}"
    )

    # Compute the FIM. Use sigma proportional to PSA (10% relative noise);
    # this models a multiplicative noise floor more representative of clinical
    # PSA assays than a uniform additive noise.
    sigma = 0.10 * np.maximum(psa_nom, 0.1 * psa_nom.max())
    fim_result = compute_fim(
        predict=predict_psa,
        theta_nominal=PARAM_NOMINAL,
        eps_rel=1e-3,
        sigma=sigma,
        param_names=PARAM_NAMES,
    )
    decomp = fim_eigendecomposition(fim_result)

    log.info(f"  Effective rank: {decomp['effective_rank']} / {decomp['n_params']}")
    log.info(f"  Rank-deficient: {decomp['rank_deficient']}")
    log.info(f"  Condition number: {decomp['condition_number']:.2e}")
    log.info(f"  Eigenvalues (desc): {decomp['eigenvalues'].tolist()}")

    # --- Figure ---
    fig = plt.figure(figsize=(13, 11))
    fig.set_facecolor("white")
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.45, wspace=0.32)

    # Panel 1: nominal PSA trajectory under MTD
    ax_psa = fig.add_subplot(gs[0, 0])
    ax_psa.plot(T_OBS, psa_nom, color="tab:purple", linewidth=1.6, marker="o", markersize=3, label="PSA(t)")
    ax_psa.fill_between(T_OBS, psa_nom - sigma, psa_nom + sigma, color="tab:purple", alpha=0.15, label="±σ (10% rel.)")
    ax_psa.set_xlabel("Time (days)")
    ax_psa.set_ylabel("PSA")
    ax_psa.set_title("Nominal PSA trajectory under MTD\n(observation schedule for FIM)", fontsize=10)
    ax_psa.legend(fontsize=9)
    ax_psa.grid(True, alpha=0.3)

    # Panel 2: FIM heatmap (log-scale)
    ax_fim = fig.add_subplot(gs[0, 1])
    fim_log = np.log10(np.abs(fim_result.fim) + 1e-20)
    im = ax_fim.imshow(fim_log, cmap="viridis", aspect="auto")
    ax_fim.set_xticks(range(len(PARAM_NAMES)))
    ax_fim.set_yticks(range(len(PARAM_NAMES)))
    ax_fim.set_xticklabels(PARAM_NAMES, rotation=45, ha="right")
    ax_fim.set_yticklabels(PARAM_NAMES)
    ax_fim.set_title("FIM (log₁₀|entries|)", fontsize=11)
    fig.colorbar(im, ax=ax_fim, fraction=0.046, pad=0.04)
    # annotate values
    for i in range(len(PARAM_NAMES)):
        for j in range(len(PARAM_NAMES)):
            ax_fim.text(
                j, i, f"{fim_log[i, j]:.1f}",
                ha="center", va="center", color="white", fontsize=8,
            )

    # Panel 3: Eigenvalue spectrum
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
                    label=f"identifiability threshold (10⁻⁶ × λ_max)")
    ax_spec.grid(True, alpha=0.3, axis="y", which="both")
    ax_spec.legend(fontsize=9)

    # Panel 4: Most-identifiable + least-identifiable directions
    ax_dir = fig.add_subplot(gs[1, 1])
    n = len(PARAM_NAMES)
    width = 0.4
    x = np.arange(n)
    most = decomp["most_identifiable_direction"]
    least = decomp["least_identifiable_direction"]
    # Sign-normalize so the largest magnitude entry is positive — for visual clarity.
    if abs(most[np.argmax(np.abs(most))]) > 0:
        most = most * np.sign(most[np.argmax(np.abs(most))])
    if abs(least[np.argmax(np.abs(least))]) > 0:
        least = least * np.sign(least[np.argmax(np.abs(least))])
    ax_dir.bar(x - width / 2, most, width, color="tab:green", label=f"most identifiable (λ_1={eigs[0]:.2g})")
    ax_dir.bar(x + width / 2, least, width, color="tab:red", label=f"least identifiable (λ_{n}={eigs[-1]:.2g})")
    ax_dir.axhline(0, color="black", linewidth=0.5)
    ax_dir.set_xticks(x)
    ax_dir.set_xticklabels(PARAM_NAMES)
    ax_dir.set_ylabel("Eigenvector component")
    ax_dir.set_title("Identifiability eigenvectors", fontsize=11)
    ax_dir.legend(fontsize=8)
    ax_dir.grid(True, alpha=0.3, axis="y")

    # Panel 5: Parameter-pair correlation matrix (FIM⁻¹ correlations)
    ax_corr = fig.add_subplot(gs[2, 0])
    # Pseudo-inverse for rank-deficient case
    fim_inv = np.linalg.pinv(fim_result.fim)
    diag_sqrt = np.sqrt(np.maximum(np.diag(fim_inv), 0))
    # Avoid zero-divide
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
            ax_corr.text(
                j, i, f"{corr[i, j]:.2f}",
                ha="center", va="center",
                color=("white" if abs(corr[i, j]) > 0.5 else "black"),
                fontsize=8,
            )

    # Panel 6: Sensitivity time series
    ax_sens = fig.add_subplot(gs[2, 1])
    for i, name in enumerate(PARAM_NAMES):
        # Normalize by max absolute sensitivity for shape comparison
        s = fim_result.sensitivities[i]
        norm = np.max(np.abs(s)) + 1e-30
        ax_sens.plot(T_OBS, s / norm, linewidth=1.4, label=f"∂PSA/∂{name} (normalized)")
    ax_sens.set_xlabel("Time (days)")
    ax_sens.set_ylabel("Normalized sensitivity")
    ax_sens.set_title("Per-parameter sensitivity over time", fontsize=11)
    ax_sens.legend(fontsize=9, loc="best")
    ax_sens.grid(True, alpha=0.3)
    ax_sens.axhline(0, color="black", linewidth=0.5)

    fig.suptitle(
        "FIM identifiability — 2-pop L-V multdeath under MTD, PSA-only observation (Stage 2.5b)",
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
    base = _REPO_ROOT / "results" / "figures" / f"fig04_fim_identifiability_{sha}_{date}"
    fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {base}.png and .pdf")

    # JSON summary
    summary = {
        "experiment": "fim_identifiability_2pop",
        "git_sha": sha,
        "date": date,
        "param_names": PARAM_NAMES,
        "param_nominal": PARAM_NOMINAL.tolist(),
        "fixed_params": {
            "K": K_FIXED, "d": D_FIXED, "rho": RHO_FIXED, "phi": PHI_FIXED,
            "S0": S0_FIXED, "R0": R0_FIXED,
        },
        "observation": {
            "schedule": "MTD only (drug always on)",
            "n_obs": len(T_OBS),
            "t_max": float(T_OBS[-1]),
            "noise_model": "10% relative (sigma = 0.10 * max(PSA, 0.1*PSA_max))",
        },
        "fim_eigenvalues": decomp["eigenvalues"].tolist(),
        "effective_rank": decomp["effective_rank"],
        "n_params": decomp["n_params"],
        "rank_deficient": bool(decomp["rank_deficient"]),
        "condition_number": float(decomp["condition_number"]),
        "most_identifiable_direction": decomp["most_identifiable_direction"].tolist(),
        "least_identifiable_direction": decomp["least_identifiable_direction"].tolist(),
        "interpretation_notes": [
            "Eigenvalues span many orders of magnitude — typical for biological L-V models.",
            "The least-identifiable direction shows which parameter combination produces near-identical PSA trajectories.",
            "A more informative observation schedule (e.g., AT50 with cycling) would change this picture substantially.",
            "This is single-schedule single-patient FIM. Phase 3 candidate C extends to per-patient FIM with multi-patient pooling.",
        ],
    }
    summary_path = (
        _REPO_ROOT / "results" / f"fim_summary_{sha}_{date}.json"
    )
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FIM identifiability analysis")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (unused for deterministic FIM)")
    args = parser.parse_args()
    main(seed=args.seed)
