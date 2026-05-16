"""Bayesian MCMC on a synthetic PSA trajectory — Phase 3 Candidate C §3.3 prototype.

Experiment 09 used the FIM-induced regularized Gaussian as a posterior surrogate.
That's an asymptotic Cramér-Rao approximation; the *actual* posterior on the
likelihood may have heavy tails, multi-modality, or boundary effects that the
Gaussian misses. This experiment runs an actual MCMC on a synthetic PSA
trajectory generated from a known theta_true, and compares the MCMC posterior
to the FIM-Gaussian.

Setup:
- Model: 3-pop K-shift Zhang, 6 fitted parameters (same as experiments 8, 9).
- theta_true = canonical Zhang parameters.
- Synthetic data: y_obs = predict_psa_under_mtd(theta_true) + N(0, sigma) noise,
  with sigma = 10% relative (same noise model as 8, 9).
- Likelihood: Gaussian residuals,
    log L(theta | y) = -0.5 sum_k ((y_k - y_pred_k(theta)) / sigma_k)^2
- Prior: improper uniform with physical constraints (positivity).
- MCMC: simple component-wise Metropolis-Hastings, adaptive step sizes
  (Roberts-Rosenthal target acceptance ~0.234 in 6 dimensions).
- N_steps = 4000 with 1500 burn-in, thinning every 4.

Output:
- MCMC samples (post-burnin, post-thinning).
- Comparison figure: marginal histograms (MCMC vs FIM-Gaussian).
- Policy comparison on a subset of MCMC samples, mirroring experiment 9.
- JSON summary.

This validates that the FIM-Gaussian approximation is reasonable in this
regime, OR surfaces deviations that motivate using actual MCMC for Phase 3.

Note on running time: ~5000 MCMC steps × ~10 ms per likelihood eval =
~50 seconds of MCMC. Plus the policy comparison at ~100 posterior samples
× 5 patients × 2 policies = ~1000 simulations × 100 ms = ~100 seconds.
Total ~3 minutes — feasible.
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
THETA_TRUE = np.array([
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
    return None  # solver fail at this theta -> log_likelihood = -inf


def is_physically_valid(theta: np.ndarray) -> bool:
    r_Tplus, r_TP, r_Tminus, alpha_2_0, alpha_2_1, K_TP_drop = theta
    if r_Tplus <= 0 or r_TP <= 0 or r_Tminus <= 0: return False
    if alpha_2_0 < 0 or alpha_2_1 < 0: return False
    if K_TP_drop < 0 or K_TP_drop > _canon.K_TP_max: return False
    return True


def log_posterior(theta: np.ndarray, y_obs: np.ndarray, sigma: np.ndarray) -> float:
    """Log posterior = log likelihood + log prior. Uniform prior with physical constraints."""
    if not is_physically_valid(theta):
        return -np.inf
    y_pred = predict_psa_under_mtd(theta)
    if y_pred is None:
        return -np.inf
    residuals = (y_obs - y_pred) / sigma
    return -0.5 * float(np.sum(residuals ** 2))


def adaptive_metropolis(
    log_post_fn,
    theta0: np.ndarray,
    n_steps: int,
    burn_in: int,
    thin: int,
    rng: np.random.Generator,
    proposal_scale_init: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Component-wise Metropolis-Hastings with adaptive step sizes during burn-in."""
    n_dim = len(theta0)
    theta = theta0.copy()
    log_p = log_post_fn(theta)
    if not np.isfinite(log_p):
        raise ValueError("Initial theta has -inf log_posterior")

    samples = []
    proposal_scale = proposal_scale_init.copy()
    accept_counter = np.zeros(n_dim)
    propose_counter = np.zeros(n_dim)
    accept_log = []

    for step in range(n_steps):
        # Cycle through dimensions, propose one at a time.
        dim = step % n_dim
        delta = rng.normal() * proposal_scale[dim]
        theta_proposed = theta.copy()
        theta_proposed[dim] = theta[dim] + delta
        log_p_proposed = log_post_fn(theta_proposed)
        log_alpha = log_p_proposed - log_p
        propose_counter[dim] += 1
        if np.log(rng.uniform()) < log_alpha:
            theta = theta_proposed
            log_p = log_p_proposed
            accept_counter[dim] += 1

        # Adapt step sizes during burn-in. Roberts-Rosenthal target ~0.234 in d>5.
        if step < burn_in and step > 0 and step % (10 * n_dim) == 0:
            for d in range(n_dim):
                if propose_counter[d] >= 5:
                    rate = accept_counter[d] / propose_counter[d]
                    if rate < 0.20:
                        proposal_scale[d] *= 0.85
                    elif rate > 0.45:
                        proposal_scale[d] *= 1.15
                    accept_counter[d] = 0
                    propose_counter[d] = 0

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(theta.copy())
        accept_log.append(log_p)

    return np.array(samples), {
        "n_total_steps": n_steps,
        "burn_in": burn_in,
        "thin": thin,
        "n_samples": len(samples),
        "final_proposal_scale": proposal_scale.tolist(),
        "log_posterior_trace": accept_log,
    }


def main(seed: int = 0, n_steps: int = 4000, burn_in: int = 1500, thin: int = 4) -> None:
    warnings.filterwarnings("ignore")
    log.info("MCMC posterior on synthetic PSA — Phase 3 Candidate C §3.3 prototype")

    # --- Generate synthetic data ---
    log.info("Step 1: generate synthetic PSA data at theta_true")
    rng = np.random.default_rng(seed)
    y_clean = predict_psa_under_mtd(THETA_TRUE)
    sigma = 0.10 * np.maximum(y_clean, 0.1 * y_clean.max())
    y_obs = y_clean + rng.normal(size=y_clean.shape) * sigma
    log.info(f"  Generated {len(y_obs)} synthetic PSA observations")

    # --- FIM-Gaussian baseline ---
    log.info("Step 2: compute FIM-Gaussian baseline (for comparison)")
    fim_result = compute_fim(
        predict=predict_psa_under_mtd,
        theta_nominal=THETA_TRUE, eps_rel=1e-3, sigma=sigma,
        param_names=PARAM_NAMES,
    )
    fim = fim_result.fim
    eigvals_fim, eigvecs_fim = np.linalg.eigh(fim)
    eigvals_reg = np.maximum(eigvals_fim, eigvals_fim.max() * 1e-3)
    cov_fim = eigvecs_fim @ np.diag(1.0 / eigvals_reg) @ eigvecs_fim.T
    cov_fim = 0.5 * (cov_fim + cov_fim.T)
    fim_std = np.sqrt(np.diag(cov_fim))
    log.info(f"  FIM-Gaussian std per param: {fim_std.tolist()}")

    # --- MCMC ---
    log.info(f"Step 3: run adaptive MH MCMC (n_steps={n_steps}, burn_in={burn_in}, thin={thin})")
    # Initialize MCMC at theta_true (we'd never know this in practice, but
    # this isolates "is the FIM-Gaussian a good approximation to the actual
    # posterior" from "can MCMC find the mode" — the latter is a separate
    # MCMC engineering question).
    # Initial proposal scale: half the FIM-Gaussian std.
    proposal_scale_init = 0.5 * fim_std
    log_post_fn = lambda th: log_posterior(th, y_obs, sigma)  # noqa: E731

    mcmc_samples, mcmc_info = adaptive_metropolis(
        log_post_fn=log_post_fn,
        theta0=THETA_TRUE.copy(),
        n_steps=n_steps,
        burn_in=burn_in,
        thin=thin,
        rng=rng,
        proposal_scale_init=proposal_scale_init,
    )
    log.info(f"  Got {len(mcmc_samples)} post-burnin thinned samples")
    log.info(f"  Final adaptive proposal scales: {mcmc_info['final_proposal_scale']}")

    # --- Compare MCMC marginal stats to FIM-Gaussian ---
    log.info("Step 4: compare marginal posteriors")
    mcmc_mean = mcmc_samples.mean(axis=0)
    mcmc_std = mcmc_samples.std(axis=0)
    log.info(f"  param        true        MCMC mean   MCMC std    FIM std    ratio (MCMC/FIM)")
    for i, name in enumerate(PARAM_NAMES):
        ratio = mcmc_std[i] / fim_std[i] if fim_std[i] > 0 else float("inf")
        log.info(
            f"  {name:10s}  {THETA_TRUE[i]:10.4g}  {mcmc_mean[i]:10.4g}  "
            f"{mcmc_std[i]:10.4g}  {fim_std[i]:10.4g}  {ratio:5.2f}"
        )

    # --- Policy comparison on subsample of MCMC samples ---
    log.info("Step 5: policy comparison on subsample of MCMC posterior")
    n_policy_subsample = min(50, len(mcmc_samples))
    subsample_idx = rng.choice(len(mcmc_samples), n_policy_subsample, replace=False)
    n_patients_per = 3
    policy_results = []
    for k, idx in enumerate(subsample_idx):
        if (k + 1) % 10 == 0:
            log.info(f"    {k+1}/{n_policy_subsample} samples scored")
        theta = mcmc_samples[idx]
        lv = _build_lv_params(theta)
        ttp_mtd, ttp_at50 = [], []
        dose_mtd, dose_at50 = [], []
        ss = np.random.SeedSequence(seed * 1000 + idx)
        children = ss.spawn(n_patients_per)
        for child in children:
            patient_rng = np.random.Generator(np.random.PCG64(child))
            params = ZhangPatientParams(lv_params=lv)
            try:
                r_mtd = run_zhang_patient(params, MTDPolicy(), rng=patient_rng)
                r_at50 = run_zhang_patient(params, AT50Policy(), rng=patient_rng)
                ttp_mtd.append(r_mtd["ttp"]); dose_mtd.append(r_mtd["cumulative_dose"])
                ttp_at50.append(r_at50["ttp"]); dose_at50.append(r_at50["cumulative_dose"])
            except Exception:  # noqa: BLE001
                continue
        if not ttp_mtd or not ttp_at50:
            continue
        policy_results.append({
            "ttp_mtd_med": float(np.median(ttp_mtd)),
            "ttp_at50_med": float(np.median(ttp_at50)),
            "drug_frac_mtd": float(np.mean(dose_mtd) / np.mean(ttp_mtd)) if np.mean(ttp_mtd) > 0 else 0.0,
            "drug_frac_at50": float(np.mean(dose_at50) / np.mean(ttp_at50)) if np.mean(ttp_at50) > 0 else 0.0,
        })
    advantages = np.array([r["ttp_at50_med"] - r["ttp_mtd_med"] for r in policy_results])
    drug_savings = np.array([
        1.0 - r["drug_frac_at50"] / max(r["drug_frac_mtd"], 1e-9) for r in policy_results
    ])
    p_at50_wins_ttp = float(np.mean(advantages > 0))
    p_at50_wins_drug = float(np.mean(drug_savings > 0))
    log.info(f"  Policy comparison on {len(policy_results)} MCMC samples × {n_patients_per} patients:")
    log.info(f"  P(AT50 wins TTP)  = {p_at50_wins_ttp:.0%}")
    log.info(f"  P(AT50 wins drug) = {p_at50_wins_drug:.0%}")
    log.info(f"  Median advantage  = {np.median(advantages):.0f} d")

    # --- Figure ---
    log.info("Step 6: figure")
    fig = plt.figure(figsize=(15, 11))
    fig.set_facecolor("white")
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.32)

    # Row 1: log-posterior trace + observed-vs-fitted PSA
    ax_trace = fig.add_subplot(gs[0, :2])
    ax_trace.plot(mcmc_info["log_posterior_trace"], color="tab:purple", linewidth=0.6, alpha=0.7)
    ax_trace.axvline(burn_in, color="tab:red", linestyle="--", linewidth=1.0, label=f"burn-in (step {burn_in})")
    ax_trace.set_xlabel("MCMC step")
    ax_trace.set_ylabel("log posterior")
    ax_trace.set_title("MCMC trace (log posterior over steps)", fontsize=11)
    ax_trace.legend(fontsize=9)
    ax_trace.grid(True, alpha=0.3)

    ax_obs = fig.add_subplot(gs[0, 2])
    ax_obs.errorbar(T_OBS, y_obs, yerr=sigma, fmt="o", color="tab:gray", markersize=3, label="y_obs (synthetic)")
    ax_obs.plot(T_OBS, y_clean, color="black", linewidth=1.5, alpha=0.6, label="y_pred(θ_true)")
    ax_obs.set_xlabel("Time (days)")
    ax_obs.set_ylabel("PSA")
    ax_obs.set_title("Synthetic data + true trajectory", fontsize=11)
    ax_obs.legend(fontsize=8)
    ax_obs.grid(True, alpha=0.3)

    # Row 2: marginal posterior comparisons (3 of 6 params per row)
    for i in range(min(6, len(PARAM_NAMES))):
        row = 1 + (i // 3)
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        # MCMC marginal histogram
        ax.hist(mcmc_samples[:, i], bins=40, density=True, alpha=0.6, color="tab:blue", label="MCMC")
        # FIM-Gaussian curve
        x_grid = np.linspace(mcmc_samples[:, i].min(), mcmc_samples[:, i].max(), 200)
        gaussian = np.exp(-0.5 * ((x_grid - THETA_TRUE[i]) / fim_std[i]) ** 2) / (fim_std[i] * np.sqrt(2 * np.pi))
        ax.plot(x_grid, gaussian, color="tab:red", linewidth=1.5, label="FIM-Gaussian")
        ax.axvline(THETA_TRUE[i], color="black", linestyle="--", linewidth=1.0, label="θ_true")
        ax.set_xlabel(PARAM_NAMES[i])
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"MCMC vs FIM-Gaussian on synthetic PSA — Phase 3 §3.3 prototype\n"
        f"P(AT50 wins TTP) = {p_at50_wins_ttp:.0%} on {len(policy_results)} MCMC samples × {n_patients_per} patients",
        fontsize=12,
    )

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True,
        ).strip()
    except subprocess.CalledProcessError:
        sha = "uncommitted"
    date = dt.date.today().isoformat()
    base = _REPO_ROOT / "results" / "figures" / f"fig10_mcmc_synthetic_psa_{sha}_{date}"
    fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {base}.png and .pdf")

    # JSON summary
    summary = {
        "experiment": "mcmc_synthetic_psa",
        "git_sha": sha,
        "date": date,
        "model": "3-pop K-shift Zhang",
        "param_names": PARAM_NAMES,
        "theta_true": THETA_TRUE.tolist(),
        "mcmc": {
            "method": "Adaptive component-wise Metropolis-Hastings",
            "n_total_steps": int(mcmc_info["n_total_steps"]),
            "burn_in": int(mcmc_info["burn_in"]),
            "thin": int(mcmc_info["thin"]),
            "n_post_burnin_samples": int(mcmc_info["n_samples"]),
            "final_proposal_scale": mcmc_info["final_proposal_scale"],
            "mcmc_mean": mcmc_mean.tolist(),
            "mcmc_std": mcmc_std.tolist(),
        },
        "fim_gaussian_std": fim_std.tolist(),
        "std_ratio_mcmc_over_fim": [
            float(mcmc_std[i] / fim_std[i]) if fim_std[i] > 0 else float("inf")
            for i in range(len(PARAM_NAMES))
        ],
        "policy_comparison": {
            "n_subsample": len(policy_results),
            "n_patients_per": n_patients_per,
            "p_at50_wins_ttp": p_at50_wins_ttp,
            "p_at50_wins_drug": p_at50_wins_drug,
            "median_advantage_d": float(np.median(advantages)) if advantages.size else None,
        },
        "interpretation_notes": [
            "If MCMC std / FIM std is close to 1 across all parameters, the FIM-Gaussian is a faithful surrogate for the actual posterior in this regime.",
            "If the ratio is much greater than 1, the actual posterior has heavier tails and/or non-Gaussian structure that the FIM misses.",
            "If the ratio is much less than 1 in identifiable directions, the FIM regularization (1e-3 floor on eigenvalues) is too generous.",
            "Phase 3 §3.3 production version: replace MCMC starting point at theta_true with a more realistic initialization (e.g., least-squares ML estimate); validate that MCMC reaches the same mode.",
        ],
    }
    summary_path = _REPO_ROOT / "results" / f"mcmc_synthetic_summary_{sha}_{date}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCMC on synthetic PSA")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=4000)
    parser.add_argument("--burn-in", type=int, default=1500)
    parser.add_argument("--thin", type=int, default=4)
    args = parser.parse_args()
    main(seed=args.seed, n_steps=args.n_steps, burn_in=args.burn_in, thin=args.thin)
