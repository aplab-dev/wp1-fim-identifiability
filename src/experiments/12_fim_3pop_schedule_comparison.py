"""Cross-schedule FIM on the 3-pop K-shift Zhang model — WP1 §3.5.

Extends experiment 05 (cross-schedule FIM on 2-pop multdeath, rank 1/4
across schedules) and experiment 08 (3-pop FIM under MTD, rank 3/6) to
test whether the 3-pop rank advantage is also schedule-invariant.

Hypothesis. The 3-pop K-shift dynamics has multiple time-scales (T+
collapse fast under K_T+ collapse, TP linear-collapse, T- slow regrowth).
The MTD schedule already excites all three, giving rank 3/6. AT50
cycling and forced-periodic schedules excite the same time-scales plus
the regrowth-during-drug-holiday phase. Question: do they expose any
NEW directions?

For WP1 §4 cross-schedule invariance to extend to the 3-pop case, we
need to find that ALL three schedules give comparable rank. If MTD and
AT50/periodic differ in rank, that's also a publishable finding —
"clinical-tribe model identifiability is schedule-dependent."

Implementation. Same 6-parameter subset as experiment 08:
(r_T+, r_TP, r_T-, alpha[2,0], alpha[2,1], K_TP_drop). Use replayed-AT50
for the cycling schedule (per experiment 05's design) — record the
schedule at nominal params, then replay for FIM perturbations to avoid
the toggle-time discontinuity.

Output:
- ``results/figures/fig12_fim_3pop_schedule_comparison_{git_sha}_{date}.{png,pdf}``
- ``results/fim_3pop_schedule_summary_{git_sha}_{date}.json``
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
_canon = zhang_canonical_lv_params()
THETA_TRUE = np.array([
    _canon.r_Tplus, _canon.r_TP, _canon.r_Tminus,
    float(_canon.alpha[2, 0]), float(_canon.alpha[2, 1]),
    _canon.K_TP_drop,
])
T_OBS = np.arange(0.0, 1500.0 + 1, 28.0)
DECISION_INTERVAL = 28.0


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


def predict_psa(theta: np.ndarray, schedule_fn) -> np.ndarray | None:
    """Simulate the 3-pop K-shift + PSA filter under given schedule, return PSA at T_OBS."""
    sim = LV3PopKShift(_build_lv_params(theta))
    psa_params = PSAParams()

    def rhs(t, y):
        x = y[:3]
        psa = y[3]
        u = schedule_fn(t)
        dx = sim.dynamics(t, x, u)
        dpsa = psa_params.rho * float(np.sum(x)) - psa_params.phi * psa
        return np.concatenate([dx, [dpsa]])

    psa0 = psa_steady_state(float(np.sum(ZHANG_CANONICAL_X0)), psa_params)
    y0 = np.array([*ZHANG_CANONICAL_X0, psa0])
    for method in ("LSODA", "BDF"):
        try:
            sol = solve_ivp(
                rhs, t_span=(0.0, T_OBS[-1]), y0=y0, t_eval=T_OBS,
                method=method, rtol=1e-8, atol=1e-3,
            )
            if sol.success:
                return sol.y[3]
        except Exception:  # noqa: BLE001
            continue
    return None


def at50_schedule_at_nominal() -> tuple[np.ndarray, np.ndarray]:
    """Run AT50 at nominal theta on the 3-pop model. Return (t_decisions, u_decisions)."""
    sim = LV3PopKShift(_build_lv_params(THETA_TRUE))
    psa_params = PSAParams()
    psa_baseline = psa_steady_state(float(np.sum(ZHANG_CANONICAL_X0)), psa_params)
    withdraw_thresh = 0.5 * psa_baseline
    resume_thresh = 1.0 * psa_baseline

    def rhs(t, y, u):
        x = y[:3]
        psa = y[3]
        dx = sim.dynamics(t, x, u)
        dpsa = psa_params.rho * float(np.sum(x)) - psa_params.phi * psa
        return np.concatenate([dx, [dpsa]])

    state = np.array([*ZHANG_CANONICAL_X0, psa_baseline])
    in_drug_phase = True
    t_decisions = [0.0]
    u_decisions = [1.0]
    t_now = 0.0
    while t_now < T_OBS[-1]:
        t_end = min(t_now + DECISION_INTERVAL, T_OBS[-1])
        u = 1.0 if in_drug_phase else 0.0
        sol = solve_ivp(
            lambda t, y, u=u: rhs(t, y, u), t_span=(t_now, t_end),
            y0=state, t_eval=np.array([t_end]),
            method="LSODA", rtol=1e-8, atol=1e-3,
        )
        if sol.success:
            state = sol.y[:, -1]
            psa_now = float(state[3])
        else:
            break
        if in_drug_phase and psa_now <= withdraw_thresh:
            in_drug_phase = False
        elif (not in_drug_phase) and psa_now >= resume_thresh:
            in_drug_phase = True
        t_decisions.append(t_end)
        u_decisions.append(1.0 if in_drug_phase else 0.0)
        t_now = t_end
    return np.array(t_decisions), np.array(u_decisions)


def _make_step_fn(t_decisions, u_decisions):
    def u_of_t(t):
        idx = int(np.searchsorted(t_decisions, t, side="right") - 1)
        idx = max(0, min(idx, len(u_decisions) - 1))
        return float(u_decisions[idx])
    return u_of_t


def main(seed: int = 0) -> None:
    warnings.filterwarnings("ignore")
    log.info("Cross-schedule FIM on 3-pop K-shift Zhang (WP1 §3.5)")

    # Schedule definitions
    mtd_schedule = lambda t: 1.0  # noqa: E731
    log.info("  Generating replayed-AT50 schedule at nominal theta...")
    t_at50, u_at50 = at50_schedule_at_nominal()
    n_toggles = int(np.sum(np.diff(u_at50) != 0))
    log.info(f"    AT50 toggles: {n_toggles}")
    at50_schedule = _make_step_fn(t_at50, u_at50)
    # Periodic 56d / 50% duty cycle (matches experiment 05 settings)
    periodic_schedule = lambda t: 1.0 if (t % 56.0) < 28.0 else 0.0  # noqa: E731

    # PSA trajectories at nominal
    log.info("  Computing nominal PSA trajectories...")
    psa_mtd_nom = predict_psa(THETA_TRUE, mtd_schedule)
    psa_at50_nom = predict_psa(THETA_TRUE, at50_schedule)
    psa_periodic_nom = predict_psa(THETA_TRUE, periodic_schedule)
    if psa_mtd_nom is None or psa_at50_nom is None or psa_periodic_nom is None:
        log.error("Failed to compute nominal trajectory under one of the schedules")
        return
    log.info(
        f"    MTD       PSA range: [{psa_mtd_nom.min():.0f}, {psa_mtd_nom.max():.0f}]\n"
        f"    AT50      PSA range: [{psa_at50_nom.min():.0f}, {psa_at50_nom.max():.0f}]\n"
        f"    Periodic  PSA range: [{psa_periodic_nom.min():.0f}, {psa_periodic_nom.max():.0f}]"
    )

    # FIM under each schedule (10% relative noise model)
    sigma_mtd = 0.10 * np.maximum(psa_mtd_nom, 0.1 * psa_mtd_nom.max())
    sigma_at50 = 0.10 * np.maximum(psa_at50_nom, 0.1 * psa_at50_nom.max())
    sigma_periodic = 0.10 * np.maximum(psa_periodic_nom, 0.1 * psa_periodic_nom.max())

    log.info("  Computing FIMs...")
    fim_mtd = compute_fim(
        predict=lambda th: predict_psa(th, mtd_schedule),
        theta_nominal=THETA_TRUE, eps_rel=1e-3, sigma=sigma_mtd, param_names=PARAM_NAMES,
    )
    fim_at50 = compute_fim(
        predict=lambda th: predict_psa(th, at50_schedule),
        theta_nominal=THETA_TRUE, eps_rel=1e-3, sigma=sigma_at50, param_names=PARAM_NAMES,
    )
    fim_periodic = compute_fim(
        predict=lambda th: predict_psa(th, periodic_schedule),
        theta_nominal=THETA_TRUE, eps_rel=1e-3, sigma=sigma_periodic, param_names=PARAM_NAMES,
    )
    decomp_mtd = fim_eigendecomposition(fim_mtd)
    decomp_at50 = fim_eigendecomposition(fim_at50)
    decomp_periodic = fim_eigendecomposition(fim_periodic)

    log.info(f"  MTD       rank = {decomp_mtd['effective_rank']}/6, "
             f"eigvals = {[f'{e:.2g}' for e in decomp_mtd['eigenvalues']]}")
    log.info(f"  AT50      rank = {decomp_at50['effective_rank']}/6, "
             f"eigvals = {[f'{e:.2g}' for e in decomp_at50['eigenvalues']]}")
    log.info(f"  Periodic  rank = {decomp_periodic['effective_rank']}/6, "
             f"eigvals = {[f'{e:.2g}' for e in decomp_periodic['eigenvalues']]}")

    # --- Figure ---
    fig = plt.figure(figsize=(16, 10))
    fig.set_facecolor("white")
    gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.32)

    # Build dense periodic schedule for shading
    t_periodic_dense = np.linspace(0, T_OBS[-1], 1500)
    u_periodic_dense = np.array([periodic_schedule(t) for t in t_periodic_dense])

    schedules_for_plot = [
        ("MTD", np.array([0, T_OBS[-1]]), np.array([1.0, 1.0]), psa_mtd_nom, "tab:blue"),
        ("Replayed AT50", t_at50, u_at50, psa_at50_nom, "tab:red"),
        ("Periodic 56d/50%", t_periodic_dense, u_periodic_dense, psa_periodic_nom, "tab:orange"),
    ]
    for col, (label, t_full, u_full, psa_nom, color) in enumerate(schedules_for_plot):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(T_OBS, psa_nom, color=color, linewidth=1.4, marker="o", markersize=2.5)
        on_ranges = []
        cur = None
        for i, u in enumerate(u_full):
            if u > 0.5 and cur is None:
                cur = float(t_full[i])
            elif u <= 0.5 and cur is not None:
                on_ranges.append((cur, float(t_full[i]))); cur = None
        if cur is not None:
            on_ranges.append((cur, float(t_full[-1])))
        for t0, t1 in on_ranges:
            ax.axvspan(t0, t1, color="tab:gray", alpha=0.12, linewidth=0)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("PSA")
        ax.set_title(f"{label} — nominal PSA + schedule", fontsize=10, color=color)
        ax.grid(True, alpha=0.3)

    # Joint eigenvalue spectrum
    ax_spec = fig.add_subplot(gs[0, 3])
    n = 6
    x = np.arange(n)
    width = 0.27
    ax_spec.bar(x - width, decomp_mtd["eigenvalues"], width, label=f"MTD (rank {decomp_mtd['effective_rank']})", color="tab:blue", edgecolor="black", linewidth=0.5)
    ax_spec.bar(x, decomp_at50["eigenvalues"], width, label=f"AT50 (rank {decomp_at50['effective_rank']})", color="tab:red", edgecolor="black", linewidth=0.5)
    ax_spec.bar(x + width, decomp_periodic["eigenvalues"], width, label=f"Periodic (rank {decomp_periodic['effective_rank']})", color="tab:orange", edgecolor="black", linewidth=0.5)
    ax_spec.set_yscale("log")
    ax_spec.set_xticks(x)
    ax_spec.set_xticklabels([f"λ_{i+1}" for i in range(n)])
    ax_spec.set_ylabel("Eigenvalue (log scale)")
    ax_spec.set_title("FIM eigenvalue spectrum (3-pop)", fontsize=11)
    ax_spec.legend(fontsize=8)
    ax_spec.grid(True, alpha=0.3, axis="y", which="both")

    # Estimate-correlation matrices for each schedule
    for col, (label, fim_res, color) in enumerate([
        ("MTD est.corr", fim_mtd, "tab:blue"),
        ("AT50 est.corr", fim_at50, "tab:red"),
        ("Periodic est.corr", fim_periodic, "tab:orange"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        fim_inv = np.linalg.pinv(fim_res.fim)
        diag_sqrt = np.sqrt(np.maximum(np.diag(fim_inv), 0))
        denom = np.outer(diag_sqrt, diag_sqrt) + 1e-30
        corr = fim_inv / denom
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1.05, vmax=1.05, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(PARAM_NAMES, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(PARAM_NAMES, fontsize=7)
        ax.set_title(label, fontsize=10, color=color)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        color=("white" if abs(corr[i, j]) > 0.5 else "black"), fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Joint least-identifiable directions
    ax_dir = fig.add_subplot(gs[1, 3])
    width = 0.27
    least_mtd = decomp_mtd["least_identifiable_direction"].copy()
    least_at50 = decomp_at50["least_identifiable_direction"].copy()
    least_per = decomp_periodic["least_identifiable_direction"].copy()
    for v in (least_mtd, least_at50, least_per):
        if abs(v[np.argmax(np.abs(v))]) > 0:
            v *= np.sign(v[np.argmax(np.abs(v))])
    ax_dir.bar(x - width, least_mtd, width, label="MTD least-id", color="tab:blue", alpha=0.7)
    ax_dir.bar(x, least_at50, width, label="AT50 least-id", color="tab:red", alpha=0.7)
    ax_dir.bar(x + width, least_per, width, label="Periodic least-id", color="tab:orange", alpha=0.7)
    ax_dir.axhline(0, color="black", linewidth=0.5)
    ax_dir.set_xticks(x)
    ax_dir.set_xticklabels(PARAM_NAMES, rotation=20, ha="right", fontsize=8)
    ax_dir.set_ylabel("Eigenvector component")
    ax_dir.set_title("Least-identifiable direction", fontsize=10)
    ax_dir.legend(fontsize=7)
    ax_dir.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"3-pop K-shift Zhang FIM under different schedules — "
        f"MTD rank {decomp_mtd['effective_rank']}/6, AT50 rank {decomp_at50['effective_rank']}/6, "
        f"Periodic rank {decomp_periodic['effective_rank']}/6 (WP1 §3.5)",
        fontsize=12,
    )

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True,
        ).strip()
    except subprocess.CalledProcessError:
        sha = "uncommitted"
    date = dt.date.today().isoformat()
    base = _REPO_ROOT / "results" / "figures" / f"fig12_fim_3pop_schedule_comparison_{sha}_{date}"
    fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {base}.png and .pdf")

    summary = {
        "experiment": "fim_3pop_schedule_comparison",
        "git_sha": sha,
        "date": date,
        "model": "3-pop K-shift Zhang canonical",
        "param_names": PARAM_NAMES,
        "param_nominal": THETA_TRUE.tolist(),
        "schedules": {
            "MTD": {
                "effective_rank": int(decomp_mtd["effective_rank"]),
                "eigenvalues": decomp_mtd["eigenvalues"].tolist(),
                "condition_number": float(decomp_mtd["condition_number"]),
                "least_identifiable_direction": least_mtd.tolist(),
            },
            "AT50_replayed": {
                "n_toggles": int(n_toggles),
                "effective_rank": int(decomp_at50["effective_rank"]),
                "eigenvalues": decomp_at50["eigenvalues"].tolist(),
                "condition_number": float(decomp_at50["condition_number"]),
                "least_identifiable_direction": least_at50.tolist(),
            },
            "Periodic_56d_50pct": {
                "effective_rank": int(decomp_periodic["effective_rank"]),
                "eigenvalues": decomp_periodic["eigenvalues"].tolist(),
                "condition_number": float(decomp_periodic["condition_number"]),
                "least_identifiable_direction": least_per.tolist(),
            },
        },
        "interpretation_notes": [
            "Same 6-parameter subset as experiment 08; comparable to experiment 05 on the 2-pop model.",
            "If all three schedules give comparable rank, the 3-pop rank-3 advantage is intrinsic to the model + observation pair (matches WP1 §4 finding for the 2-pop case extended).",
            "If AT50 / periodic differ from MTD in rank, the 3-pop case shows schedule-dependent identifiability — a more nuanced story for WP1.",
        ],
    }
    summary_path = _REPO_ROOT / "results" / f"fim_3pop_schedule_summary_{sha}_{date}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-schedule FIM on 3-pop K-shift Zhang")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (unused for deterministic FIM)")
    args = parser.parse_args()
    main(seed=args.seed)
