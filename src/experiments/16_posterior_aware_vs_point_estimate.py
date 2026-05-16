"""M8 deliverable — posterior-aware vs point-estimate clinical decision comparison.

This is the punchline experiment for Phase 3 Candidate C. The premise:

CLINICAL SCENARIO. A clinician has a patient with mCRPC. They run a Bayesian
fit on the patient's PSA trajectory. They want to choose between MTD and
AT50. There are two ways to use the fit:

1. **Point-estimate optimal** — use the posterior mean (or MAP) as the
   "best estimate" of the patient's parameters, simulate both policies
   under that point, recommend whichever wins TTP.
2. **Posterior-aware optimal** — sample many draws from the posterior,
   simulate both policies under each, recommend whichever has the higher
   *expected* TTP across the posterior.

For most patients these agree. But in posterior-sensitive regimes
(experiment 13 finding), they CAN disagree — and posterior-aware is the
provably-correct Bayesian-decision-theoretic answer.

This experiment quantifies the disagreement. We:

1. Generate a synthetic cohort with patients sampled from a regime that
   spans posterior-sensitive (K_TP_drop=1000) to AT50-dominant (K=9900).
2. Per patient: compute FIM-induced posterior (faster than MCMC for this
   demo), draw 30 samples.
3. Compute point-estimate recommendation: which policy wins TTP at the
   posterior mean.
4. Compute posterior-aware recommendation: argmax_{π} E_θ[TTP under π].
5. For each patient: also compute the *true-parameter* recommendation
   (oracle, with the synthetic-true theta).
6. Tabulate agreement: how often does each recommendation match the
   oracle, and where do point-estimate and posterior-aware disagree?

Output:
- ``results/figures/fig16_decision_comparison_{git_sha}_{date}.{png,pdf}``
- ``results/decision_comparison_summary_{git_sha}_{date}.json``
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


_canon = zhang_canonical_lv_params()
_NOMINAL = np.array([
    _canon.r_Tplus, _canon.r_TP, _canon.r_Tminus,
    float(_canon.alpha[2, 0]), float(_canon.alpha[2, 1]),
    _canon.K_TP_drop,
])
PARAM_NAMES = ["r_T+", "r_TP", "r_T-", "α(T-,T+)", "α(T-,TP)", "K_TP_drop"]
T_OBS = np.arange(0.0, 1500.0 + 1, 28.0)


def _build_lv_params(theta: np.ndarray) -> LV3PopParams:
    alpha = _canon.alpha.copy()
    alpha[2, 0] = max(theta[3], 0.01)
    alpha[2, 1] = max(theta[4], 0.01)
    return LV3PopParams(
        r_Tplus=max(theta[0], 1e-6), r_TP=max(theta[1], 1e-6), r_Tminus=max(theta[2], 1e-6),
        K_Tminus=_canon.K_Tminus, K_TP_max=_canon.K_TP_max,
        K_TP_drop=max(min(theta[5], _canon.K_TP_max - 1), 1.0),
        mu_max=_canon.mu_max, mu_drop=_canon.mu_drop,
        alpha=alpha,
    )


def predict_psa_under_mtd(theta: np.ndarray) -> np.ndarray | None:
    sim = LV3PopKShift(_build_lv_params(theta))
    psa_params = PSAParams()

    def rhs(t, y):
        x = y[:3]; psa = y[3]
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


def expected_ttp_for_policy(theta: np.ndarray, policy_factory, n_patients: int = 3,
                    rng: np.random.Generator | None = None) -> float | None:
    """Expected TTP for a given theta under a given policy. Returns mean over n_patients."""
    if rng is None:
        rng = np.random.default_rng(0)
    try:
        params = ZhangPatientParams(lv_params=_build_lv_params(theta))
    except Exception:  # noqa: BLE001
        return None
    ttps = []
    ss = np.random.SeedSequence(int(rng.integers(0, 1_000_000)))
    for child in ss.spawn(n_patients):
        prng = np.random.Generator(np.random.PCG64(child))
        try:
            r = run_zhang_patient(params, policy_factory(), rng=prng)
            ttps.append(r["ttp"])
        except Exception:  # noqa: BLE001
            continue
    return float(np.mean(ttps)) if ttps else None


def evaluate_patient_decision(theta_true: np.ndarray, n_posterior: int = 30,
                              n_patients: int = 3, rng: np.random.Generator | None = None) -> dict | None:
    """For one patient with theta_true:
    1. Compute oracle recommendation (using theta_true).
    2. Compute FIM at theta_true → regularized Gaussian posterior.
    3. Sample posterior; compute point-estimate (posterior mean) and posterior-aware recommendations.
    Return all three + agreement flags.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # 1) Oracle
    ttp_mtd_true = expected_ttp_for_policy(theta_true, MTDPolicy, n_patients=n_patients, rng=rng)
    ttp_at_true = expected_ttp_for_policy(theta_true, AT50Policy, n_patients=n_patients, rng=rng)
    if ttp_mtd_true is None or ttp_at_true is None:
        return None
    oracle_choice = "AT50" if ttp_at_true > ttp_mtd_true else "MTD"

    # 2) FIM at theta_true → posterior
    psa_nom = predict_psa_under_mtd(theta_true)
    if psa_nom is None:
        return None
    sigma = 0.10 * np.maximum(psa_nom, 0.1 * psa_nom.max())
    fim = compute_fim(predict_psa_under_mtd, theta_true, eps_rel=1e-3, sigma=sigma).fim
    eigvals, eigvecs = np.linalg.eigh(fim)
    eigvals_reg = np.maximum(eigvals, eigvals.max() * 1e-3)
    cov = eigvecs @ np.diag(1.0 / eigvals_reg) @ eigvecs.T
    cov = 0.5 * (cov + cov.T)
    L = np.linalg.cholesky(cov + 1e-8 * np.eye(6))
    raw_draws = theta_true[None, :] + (rng.normal(size=(n_posterior, 6)) @ L.T)
    valid_draws = [d for d in raw_draws if is_physically_valid(d)]
    if not valid_draws:
        return None
    posterior_mean = np.mean(valid_draws, axis=0)

    # 3) Point-estimate decision (using posterior mean as if it were truth)
    ttp_mtd_pe = expected_ttp_for_policy(posterior_mean, MTDPolicy, n_patients=n_patients, rng=rng)
    ttp_at_pe = expected_ttp_for_policy(posterior_mean, AT50Policy, n_patients=n_patients, rng=rng)
    if ttp_mtd_pe is None or ttp_at_pe is None:
        return None
    point_estimate_choice = "AT50" if ttp_at_pe > ttp_mtd_pe else "MTD"

    # 4) Posterior-aware decision: E_θ[TTP_π] for π in {MTD, AT50}
    ttp_mtd_per = []
    ttp_at_per = []
    for draw in valid_draws:
        m = expected_ttp_for_policy(draw, MTDPolicy, n_patients=n_patients, rng=rng)
        a = expected_ttp_for_policy(draw, AT50Policy, n_patients=n_patients, rng=rng)
        if m is not None and a is not None:
            ttp_mtd_per.append(m)
            ttp_at_per.append(a)
    if not ttp_mtd_per:
        return None
    expected_ttp_mtd = float(np.mean(ttp_mtd_per))
    expected_ttp_at50 = float(np.mean(ttp_at_per))
    posterior_aware_choice = "AT50" if expected_ttp_at50 > expected_ttp_mtd else "MTD"

    return {
        "oracle_choice": oracle_choice,
        "oracle_ttp_mtd": ttp_mtd_true,
        "oracle_ttp_at50": ttp_at_true,
        "point_estimate_choice": point_estimate_choice,
        "pe_ttp_mtd": ttp_mtd_pe,
        "pe_ttp_at50": ttp_at_pe,
        "posterior_aware_choice": posterior_aware_choice,
        "pa_expected_ttp_mtd": expected_ttp_mtd,
        "pa_expected_ttp_at50": expected_ttp_at50,
        "n_posterior_valid": len(valid_draws),
        "agreement_oracle_pe": oracle_choice == point_estimate_choice,
        "agreement_oracle_pa": oracle_choice == posterior_aware_choice,
        "agreement_pe_pa": point_estimate_choice == posterior_aware_choice,
        "K_TP_drop_true": float(theta_true[5]),
        "alpha_T_minus_T_plus_true": float(theta_true[3]),
    }


def main(seed: int = 0, n_patients_eval: int = 30, n_posterior: int = 25,
         n_patients_per_sim: int = 2) -> None:
    warnings.filterwarnings("ignore")
    log.info(f"M8 — posterior-aware vs point-estimate decision comparison")
    log.info(f"  N patients to evaluate: {n_patients_eval}")
    log.info(f"  N posterior draws per patient: {n_posterior}")
    log.info(f"  N simulated patients per (theta, policy): {n_patients_per_sim}")

    # Sample patients with theta_true spanning the regime sensitivity space.
    # Mix of canonical (K=9900) and posterior-sensitive (K=1000-3000) regimes.
    rng = np.random.default_rng(seed)
    K_TP_grid_for_patients = rng.choice(
        [1000, 1500, 2500, 4000, 6000, 8000, 9500],
        size=n_patients_eval, replace=True,
    )
    alpha_grid_for_patients = rng.uniform(2.0, 5.0, size=n_patients_eval)
    thetas_true = []
    for K, alph in zip(K_TP_grid_for_patients, alpha_grid_for_patients):
        theta = _NOMINAL.copy()
        theta[5] = float(K)
        theta[3] = float(alph)
        thetas_true.append(theta)

    log.info(f"  Sampled {len(thetas_true)} patients across K_TP_drop range and alpha range")

    # Evaluate each patient
    results = []
    for i, theta in enumerate(thetas_true):
        try:
            r = evaluate_patient_decision(theta, n_posterior=n_posterior,
                                          n_patients=n_patients_per_sim, rng=rng)
            if r is not None:
                results.append(r)
            if (i + 1) % 5 == 0:
                log.info(f"    {i+1}/{len(thetas_true)} patients evaluated")
        except Exception as e:  # noqa: BLE001
            log.warning(f"    patient {i}: failed: {e}")
            continue

    if not results:
        log.error("No patient decisions evaluated.")
        return

    # Aggregate
    n_total = len(results)
    n_oracle_at50 = sum(1 for r in results if r["oracle_choice"] == "AT50")
    n_oracle_mtd = n_total - n_oracle_at50
    n_pe_at50 = sum(1 for r in results if r["point_estimate_choice"] == "AT50")
    n_pa_at50 = sum(1 for r in results if r["posterior_aware_choice"] == "AT50")
    n_agree_oracle_pe = sum(1 for r in results if r["agreement_oracle_pe"])
    n_agree_oracle_pa = sum(1 for r in results if r["agreement_oracle_pa"])
    n_disagree_pe_pa = sum(1 for r in results if not r["agreement_pe_pa"])

    log.info(f"  Oracle: {n_oracle_at50}/{n_total} AT50, {n_oracle_mtd}/{n_total} MTD")
    log.info(f"  Point-estimate accuracy vs oracle:  {n_agree_oracle_pe}/{n_total} = {n_agree_oracle_pe/n_total:.0%}")
    log.info(f"  Posterior-aware accuracy vs oracle: {n_agree_oracle_pa}/{n_total} = {n_agree_oracle_pa/n_total:.0%}")
    log.info(f"  PE vs PA disagreement: {n_disagree_pe_pa}/{n_total} = {n_disagree_pe_pa/n_total:.0%}")

    # --- Figure ---
    fig = plt.figure(figsize=(15, 10))
    fig.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.32)

    # Panel 1: scatter of oracle TTP advantage vs K_TP_drop
    ax1 = fig.add_subplot(gs[0, 0])
    K_vals = np.array([r["K_TP_drop_true"] for r in results])
    oracle_adv = np.array([r["oracle_ttp_at50"] - r["oracle_ttp_mtd"] for r in results]) / 30
    pe_adv = np.array([r["pe_ttp_at50"] - r["pe_ttp_mtd"] for r in results]) / 30
    pa_adv = np.array([r["pa_expected_ttp_at50"] - r["pa_expected_ttp_mtd"] for r in results]) / 30
    ax1.scatter(K_vals, oracle_adv, alpha=0.5, label="oracle (truth)", color="black", marker="o", s=40)
    ax1.scatter(K_vals + 50, pe_adv, alpha=0.5, label="point-estimate", color="tab:blue", marker="s", s=40)
    ax1.scatter(K_vals - 50, pa_adv, alpha=0.5, label="posterior-aware", color="tab:red", marker="^", s=40)
    ax1.axhline(0, color="black", linewidth=0.7)
    ax1.set_xlabel("K_TP_drop (true)")
    ax1.set_ylabel("AT50 - MTD TTP advantage (months)")
    ax1.set_title("Three estimates of AT50 advantage", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: agreement matrix as bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    cats = ["Oracle: AT50\n(truth says AT)", "Oracle: MTD\n(truth says MTD)"]
    pe_correct = [
        sum(1 for r in results if r["oracle_choice"] == "AT50" and r["agreement_oracle_pe"]),
        sum(1 for r in results if r["oracle_choice"] == "MTD" and r["agreement_oracle_pe"]),
    ]
    pe_wrong = [
        sum(1 for r in results if r["oracle_choice"] == "AT50" and not r["agreement_oracle_pe"]),
        sum(1 for r in results if r["oracle_choice"] == "MTD" and not r["agreement_oracle_pe"]),
    ]
    pa_correct = [
        sum(1 for r in results if r["oracle_choice"] == "AT50" and r["agreement_oracle_pa"]),
        sum(1 for r in results if r["oracle_choice"] == "MTD" and r["agreement_oracle_pa"]),
    ]
    pa_wrong = [
        sum(1 for r in results if r["oracle_choice"] == "AT50" and not r["agreement_oracle_pa"]),
        sum(1 for r in results if r["oracle_choice"] == "MTD" and not r["agreement_oracle_pa"]),
    ]
    x = np.arange(len(cats))
    width = 0.35
    ax2.bar(x - width / 2, pe_correct, width, label="PE correct", color="tab:blue", alpha=0.8)
    ax2.bar(x - width / 2, pe_wrong, width, bottom=pe_correct, label="PE wrong", color="tab:blue", alpha=0.3, hatch="//")
    ax2.bar(x + width / 2, pa_correct, width, label="PA correct", color="tab:red", alpha=0.8)
    ax2.bar(x + width / 2, pa_wrong, width, bottom=pa_correct, label="PA wrong", color="tab:red", alpha=0.3, hatch="//")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cats)
    ax2.set_ylabel("Patient count")
    ax2.set_title("Decision accuracy: PE vs PA vs Oracle", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: PE vs PA scatter (highlight disagreements)
    ax3 = fig.add_subplot(gs[0, 2])
    same_choice = [r["agreement_pe_pa"] for r in results]
    pe_choice_at50 = np.array([r["point_estimate_choice"] == "AT50" for r in results])
    pa_choice_at50 = np.array([r["posterior_aware_choice"] == "AT50" for r in results])
    colors = []
    for r in results:
        if not r["agreement_pe_pa"]:
            colors.append("tab:orange")  # disagreement
        elif r["point_estimate_choice"] == "AT50":
            colors.append("tab:red")
        else:
            colors.append("tab:blue")
    ax3.scatter(pe_adv, pa_adv, c=colors, s=40, alpha=0.7, edgecolor="none")
    diag_min = min(pe_adv.min(), pa_adv.min())
    diag_max = max(pe_adv.max(), pa_adv.max())
    ax3.plot([diag_min, diag_max], [diag_min, diag_max], "k--", alpha=0.4, linewidth=0.7, label="PE = PA")
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.axvline(0, color="black", linewidth=0.5)
    ax3.set_xlabel("Point-estimate AT50 advantage (months)")
    ax3.set_ylabel("Posterior-aware AT50 advantage (months)")
    ax3.set_title(f"PE vs PA (orange = disagreement, n={n_disagree_pe_pa})", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: K_TP_drop distribution colored by agreement
    ax4 = fig.add_subplot(gs[1, 0])
    K_disagree = K_vals[~np.array(same_choice, dtype=bool)]
    K_agree = K_vals[np.array(same_choice, dtype=bool)]
    ax4.hist([K_agree, K_disagree], bins=10, stacked=True, label=["PE = PA", "PE ≠ PA"],
             color=["tab:gray", "tab:orange"], edgecolor="black")
    ax4.set_xlabel("K_TP_drop (true)")
    ax4.set_ylabel("Patient count")
    ax4.set_title("Where do PE and PA disagree?", fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis="y")

    # Panel 5: summary text
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis("off")
    summary_text = (
        f"M8 DECISION COMPARISON\n\n"
        f"Patients evaluated:    {n_total}\n"
        f"K_TP_drop range:       [{K_vals.min():.0f}, {K_vals.max():.0f}]\n"
        f"alpha[T-,T+] range:    [{min(r['alpha_T_minus_T_plus_true'] for r in results):.1f}, "
        f"{max(r['alpha_T_minus_T_plus_true'] for r in results):.1f}]\n\n"
        f"Oracle (truth):\n"
        f"  AT50 better:  {n_oracle_at50}/{n_total} = {n_oracle_at50/n_total:.0%}\n"
        f"  MTD  better:  {n_oracle_mtd}/{n_total} = {n_oracle_mtd/n_total:.0%}\n\n"
        f"Point-estimate accuracy vs oracle:\n"
        f"  {n_agree_oracle_pe}/{n_total} = {n_agree_oracle_pe/n_total:.0%}\n\n"
        f"Posterior-aware accuracy vs oracle:\n"
        f"  {n_agree_oracle_pa}/{n_total} = {n_agree_oracle_pa/n_total:.0%}\n\n"
        f"PE vs PA disagreement: {n_disagree_pe_pa}/{n_total} = {n_disagree_pe_pa/n_total:.0%}\n\n"
        f"Phase 3 §3.2 / M8 deliverable.\n"
        f"WP4 §results candidate."
    )
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
             fontsize=9, verticalalignment="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    # Panel 6: K_TP_drop vs oracle accuracy by recommendation type
    ax6 = fig.add_subplot(gs[1, 2])
    K_bins = np.linspace(K_vals.min(), K_vals.max(), 6)
    pe_acc_bins = []
    pa_acc_bins = []
    bin_centers = []
    for k in range(len(K_bins) - 1):
        mask = (K_vals >= K_bins[k]) & (K_vals < K_bins[k + 1])
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            pe_acc_bins.append(np.nan); pa_acc_bins.append(np.nan)
        else:
            pe_acc_bins.append(np.mean([r["agreement_oracle_pe"] for r, m in zip(results, mask) if m]))
            pa_acc_bins.append(np.mean([r["agreement_oracle_pa"] for r, m in zip(results, mask) if m]))
        bin_centers.append(0.5 * (K_bins[k] + K_bins[k + 1]))
    ax6.plot(bin_centers, pe_acc_bins, color="tab:blue", linewidth=1.5, marker="s", label="PE accuracy")
    ax6.plot(bin_centers, pa_acc_bins, color="tab:red", linewidth=1.5, marker="^", label="PA accuracy")
    ax6.set_xlabel("K_TP_drop bin center")
    ax6.set_ylabel("Decision accuracy vs oracle")
    ax6.set_title("Decision accuracy across regime", fontsize=10)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(-0.05, 1.05)

    fig.suptitle(
        f"M8 — Posterior-aware vs point-estimate clinical decision (Phase 3 Candidate C punchline)",
        fontsize=12,
    )

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True,
        ).strip()
    except subprocess.CalledProcessError:
        sha = "uncommitted"
    date = dt.date.today().isoformat()
    base = _REPO_ROOT / "results" / "figures" / f"fig16_decision_comparison_{sha}_{date}"
    fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {base}.png and .pdf")

    summary = {
        "experiment": "decision_comparison_pe_vs_pa",
        "git_sha": sha,
        "date": date,
        "n_patients_evaluated": int(n_total),
        "n_posterior_per_patient": int(n_posterior),
        "n_simulated_patients_per_theta": int(n_patients_per_sim),
        "headline": {
            "oracle_at50_count": int(n_oracle_at50),
            "oracle_mtd_count": int(n_oracle_mtd),
            "pe_accuracy_vs_oracle": n_agree_oracle_pe / n_total,
            "pa_accuracy_vs_oracle": n_agree_oracle_pa / n_total,
            "pe_vs_pa_disagreement_rate": n_disagree_pe_pa / n_total,
            "pa_advantage_over_pe": (n_agree_oracle_pa - n_agree_oracle_pe) / n_total,
        },
        "per_patient_results": results,
        "interpretation_notes": [
            "Oracle: knows true theta; chooses argmax over policies of true expected TTP.",
            "Point-estimate: chooses argmax under posterior mean theta as if it were truth.",
            "Posterior-aware: chooses argmax of E_θ[TTP] integrated over the posterior.",
            "PA - PE accuracy difference quantifies the value of doing posterior-aware control.",
            "If pa_advantage_over_pe > 0, posterior-aware control is provably better in this cohort.",
            "Phase 3 §3.2 / M8 deliverable for WP4.",
        ],
    }
    summary_path = _REPO_ROOT / "results" / f"decision_comparison_summary_{sha}_{date}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-patients", type=int, default=30, dest="n_patients_eval")
    parser.add_argument("--n-posterior", type=int, default=25)
    parser.add_argument("--n-patients-per-sim", type=int, default=2)
    args = parser.parse_args()
    main(seed=args.seed, n_patients_eval=args.n_patients_eval,
         n_posterior=args.n_posterior, n_patients_per_sim=args.n_patients_per_sim)
