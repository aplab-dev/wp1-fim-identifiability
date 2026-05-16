"""FIM identifiability — MTD vs replayed-AT50 schedule comparison.

Hypothesis: a cycling schedule (AT50) exposes MORE parameter directions than
a constant schedule (MTD), because the on-off toggling resolves drug-on-phase
dynamics (governed by r_S, d) separately from drug-off-phase dynamics
(governed by r_R, alpha, beta).

The experiment computes the FIM under two schedules:
1. **MTD:** u(t) = 1.0 for all t. (Same as Stage 2.5b.)
2. **Replayed AT50:** record the u(t) schedule that AT50 produces at nominal
   parameters; then for FIM perturbations, REPLAY that fixed schedule rather
   than re-deriving u(t) from each perturbed PSA trajectory.

Why "replayed" and not "live": if AT50 toggles based on each-perturbation's
PSA, the toggle TIMES shift slightly with each parameter perturbation, and
central differences of trajectories with shifting discontinuities give noisy
sensitivities. The replayed-schedule approach asks "given THIS exact drug
schedule, how identifiable are the parameters?" — which is the meaningful
question for assay design anyway.

Output:
- ``results/figures/fig05_fim_schedule_comparison_{git_sha}_{date}.{png,pdf}``
- ``results/fim_schedule_summary_{git_sha}_{date}.json``
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
K_FIXED = 1.0
D_FIXED = 1.5
RHO_FIXED = 1.0
PHI_FIXED = 0.5
S0_FIXED = 0.6
R0_FIXED = 0.006

T_MAX = 500.0
T_OBS = np.arange(0.0, T_MAX + 1, 28.0)  # 4-week labs
DECISION_INTERVAL = 14.0  # AT50 decisions every 14 days


def at50_schedule_at_nominal_params() -> tuple[np.ndarray, np.ndarray]:
    """Run AT50 once at the NOMINAL parameter vector. Return (t_decisions,
    u_decisions) — a step-function schedule we can replay later."""
    r_S, r_R, alpha, beta = PARAM_NOMINAL
    psa_baseline = RHO_FIXED * (S0_FIXED + R0_FIXED) / PHI_FIXED
    withdraw_thresh = 0.5 * psa_baseline
    resume_thresh = 1.0 * psa_baseline

    def rhs(t: float, y: np.ndarray, u: float) -> np.ndarray:
        S, R, PSA = y
        dS = r_S * S * (1 - (S + alpha * R) / K_FIXED) - D_FIXED * u * S
        dR = r_R * R * (1 - (R + beta * S) / K_FIXED)
        dPSA = RHO_FIXED * (S + R) - PHI_FIXED * PSA
        return np.array([dS, dR, dPSA])

    state = np.array([S0_FIXED, R0_FIXED, psa_baseline])
    in_drug_phase = True
    t_decisions = [0.0]
    u_decisions = [1.0]
    t_now = 0.0
    while t_now < T_MAX:
        t_end = min(t_now + DECISION_INTERVAL, T_MAX)
        u = 1.0 if in_drug_phase else 0.0
        sol = solve_ivp(
            lambda t, y, u=u: rhs(t, y, u),
            t_span=(t_now, t_end),
            y0=state,
            t_eval=np.array([t_end]),
            method="LSODA",
            rtol=1e-8, atol=1e-10,
        )
        state = sol.y[:, -1]
        psa_now = float(state[2])
        if in_drug_phase and psa_now <= withdraw_thresh:
            in_drug_phase = False
        elif (not in_drug_phase) and psa_now >= resume_thresh:
            in_drug_phase = True
        t_decisions.append(t_end)
        u_decisions.append(1.0 if in_drug_phase else 0.0)
        t_now = t_end
    return np.array(t_decisions), np.array(u_decisions)


def _make_step_fn(t_decisions: np.ndarray, u_decisions: np.ndarray):
    """Return u(t) for a step-function schedule (left-continuous)."""

    def u_of_t(t: float) -> float:
        # Use rightmost decision boundary at-or-before t.
        idx = int(np.searchsorted(t_decisions, t, side="right") - 1)
        idx = max(0, min(idx, len(u_decisions) - 1))
        return float(u_decisions[idx])

    return u_of_t


def predict_psa(theta: np.ndarray, schedule_fn) -> np.ndarray:
    """Simulate (S, R, PSA) under the given schedule_fn(t) and return PSA at T_OBS."""
    r_S, r_R, alpha, beta = theta

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        S, R, PSA = y
        u = schedule_fn(t)
        dS = r_S * S * (1 - (S + alpha * R) / K_FIXED) - D_FIXED * u * S
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
        rtol=1e-8, atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"FIM predictor solve_ivp failed: {sol.message}")
    return sol.y[2]


def periodic_schedule(period_days: float = 56.0, duty_cycle: float = 0.5):
    """Periodic on-off schedule: drug ON for the first duty_cycle*period
    of each period, OFF for the rest. Guarantees multiple toggles regardless
    of dynamics."""

    def u_of_t(t: float) -> float:
        phase = (t % period_days) / period_days
        return 1.0 if phase < duty_cycle else 0.0

    return u_of_t


def main() -> None:
    log.info("Cross-schedule FIM comparison: MTD vs replayed-AT50 vs periodic-56d")

    # Schedule definitions
    mtd_schedule = lambda t: 1.0  # noqa: E731
    t_at50, u_at50 = at50_schedule_at_nominal_params()
    log.info(f"  AT50 toggles: {int(np.sum(np.diff(u_at50) != 0))} times")
    at50_schedule = _make_step_fn(t_at50, u_at50)
    periodic = periodic_schedule(period_days=56.0, duty_cycle=0.5)

    # PSA trajectories at nominal
    psa_mtd_nom = predict_psa(PARAM_NOMINAL, mtd_schedule)
    psa_at50_nom = predict_psa(PARAM_NOMINAL, at50_schedule)
    psa_periodic_nom = predict_psa(PARAM_NOMINAL, periodic)
    log.info(
        f"  Nominal PSA — MTD min={psa_mtd_nom.min():.3f} max={psa_mtd_nom.max():.3f} | "
        f"AT50 min={psa_at50_nom.min():.3f} max={psa_at50_nom.max():.3f} | "
        f"Periodic min={psa_periodic_nom.min():.3f} max={psa_periodic_nom.max():.3f}"
    )

    # FIM under each schedule.
    sigma_mtd = 0.10 * np.maximum(psa_mtd_nom, 0.1 * psa_mtd_nom.max())
    sigma_at50 = 0.10 * np.maximum(psa_at50_nom, 0.1 * psa_at50_nom.max())
    sigma_periodic = 0.10 * np.maximum(psa_periodic_nom, 0.1 * psa_periodic_nom.max())

    fim_mtd = compute_fim(
        predict=lambda th: predict_psa(th, mtd_schedule),
        theta_nominal=PARAM_NOMINAL, eps_rel=1e-3, sigma=sigma_mtd,
        param_names=PARAM_NAMES,
    )
    fim_at50 = compute_fim(
        predict=lambda th: predict_psa(th, at50_schedule),
        theta_nominal=PARAM_NOMINAL, eps_rel=1e-3, sigma=sigma_at50,
        param_names=PARAM_NAMES,
    )
    fim_periodic = compute_fim(
        predict=lambda th: predict_psa(th, periodic),
        theta_nominal=PARAM_NOMINAL, eps_rel=1e-3, sigma=sigma_periodic,
        param_names=PARAM_NAMES,
    )
    decomp_mtd = fim_eigendecomposition(fim_mtd)
    decomp_at50 = fim_eigendecomposition(fim_at50)
    decomp_periodic = fim_eigendecomposition(fim_periodic)

    log.info(f"  MTD       rank = {decomp_mtd['effective_rank']}/4, eigvals = {decomp_mtd['eigenvalues'].tolist()}")
    log.info(f"  AT50      rank = {decomp_at50['effective_rank']}/4, eigvals = {decomp_at50['eigenvalues'].tolist()}")
    log.info(f"  Periodic  rank = {decomp_periodic['effective_rank']}/4, eigvals = {decomp_periodic['eigenvalues'].tolist()}")

    # --- Figure ---
    fig = plt.figure(figsize=(16, 11))
    fig.set_facecolor("white")
    gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.32)

    # Build a dense periodic schedule sample for shading
    t_periodic_dense = np.linspace(0, T_MAX, 1000)
    u_periodic_dense = np.array([periodic(t) for t in t_periodic_dense])

    # Row 1: nominal PSA trajectories with drug-on shading
    for col, (label, t_full, u_full, psa_nom, color) in enumerate([
        ("MTD",
         np.array([0.0, T_MAX]), np.array([1.0, 1.0]),
         psa_mtd_nom, "tab:blue"),
        ("Replayed AT50",
         t_at50, u_at50,
         psa_at50_nom, "tab:red"),
        ("Periodic 56d/50%",
         t_periodic_dense, u_periodic_dense,
         psa_periodic_nom, "tab:orange"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(T_OBS, psa_nom, color=color, linewidth=1.6, marker="o", markersize=3)
        # Drug-on shading
        on_ranges = []
        cur_start = None
        for i, u in enumerate(u_full):
            if u > 0.5 and cur_start is None:
                cur_start = float(t_full[i])
            elif u <= 0.5 and cur_start is not None:
                on_ranges.append((cur_start, float(t_full[i])))
                cur_start = None
        if cur_start is not None:
            on_ranges.append((cur_start, float(t_full[-1])))
        for t0, t1 in on_ranges:
            ax.axvspan(t0, t1, color="tab:gray", alpha=0.12, linewidth=0)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("PSA")
        ax.set_title(f"{label} — nominal PSA + schedule", fontsize=11, color=color)
        ax.grid(True, alpha=0.3)

    # Row 1 col 3: eigenvalue spectrum three-way bars
    ax_spec = fig.add_subplot(gs[0, 3])
    n = 4
    x = np.arange(n)
    width = 0.27
    ax_spec.bar(x - width, decomp_mtd["eigenvalues"], width, label=f"MTD (rank {decomp_mtd['effective_rank']})", color="tab:blue", edgecolor="black", linewidth=0.5)
    ax_spec.bar(x, decomp_at50["eigenvalues"], width, label=f"AT50 (rank {decomp_at50['effective_rank']})", color="tab:red", edgecolor="black", linewidth=0.5)
    ax_spec.bar(x + width, decomp_periodic["eigenvalues"], width, label=f"Periodic (rank {decomp_periodic['effective_rank']})", color="tab:orange", edgecolor="black", linewidth=0.5)
    ax_spec.set_yscale("log")
    ax_spec.set_xticks(x)
    ax_spec.set_xticklabels([f"λ_{i+1}" for i in range(n)])
    ax_spec.set_ylabel("Eigenvalue (log scale)")
    ax_spec.set_title("FIM eigenvalue spectrum", fontsize=11)
    ax_spec.legend(fontsize=8)
    ax_spec.grid(True, alpha=0.3, axis="y", which="both")

    # Row 2: estimate-correlation matrices three-way
    for col, (label, fim_res, color) in enumerate([
        ("MTD estimate correlation", fim_mtd, "tab:blue"),
        ("AT50 estimate correlation", fim_at50, "tab:red"),
        ("Periodic estimate correlation", fim_periodic, "tab:orange"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        fim_inv = np.linalg.pinv(fim_res.fim)
        diag_sqrt = np.sqrt(np.maximum(np.diag(fim_inv), 0))
        denom = np.outer(diag_sqrt, diag_sqrt) + 1e-30
        corr = fim_inv / denom
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1.05, vmax=1.05, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(PARAM_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(PARAM_NAMES)
        ax.set_title(label, fontsize=11, color=color)
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i, f"{corr[i, j]:.2f}",
                    ha="center", va="center",
                    color=("white" if abs(corr[i, j]) > 0.5 else "black"),
                    fontsize=8,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 2 col 4: least-identifiable directions per schedule (the
    # informative comparison: which parameter combo stays unidentifiable?)
    ax_dir = fig.add_subplot(gs[1, 3])
    width = 0.27
    x = np.arange(n)
    least_mtd = decomp_mtd["least_identifiable_direction"]
    least_at50 = decomp_at50["least_identifiable_direction"]
    least_periodic = decomp_periodic["least_identifiable_direction"]
    # Sign-normalize for visual clarity (largest abs entry positive)
    for v in (least_mtd, least_at50, least_periodic):
        if abs(v[np.argmax(np.abs(v))]) > 0:
            v *= np.sign(v[np.argmax(np.abs(v))])

    ax_dir.bar(x - width, least_mtd, width, label="MTD least-id", color="tab:blue", alpha=0.7)
    ax_dir.bar(x, least_at50, width, label="AT50 least-id", color="tab:red", alpha=0.7)
    ax_dir.bar(x + width, least_periodic, width, label="Periodic least-id", color="tab:orange", alpha=0.7)
    ax_dir.axhline(0, color="black", linewidth=0.5)
    ax_dir.set_xticks(x)
    ax_dir.set_xticklabels(PARAM_NAMES)
    ax_dir.set_ylabel("Eigenvector component")
    ax_dir.set_title("Least-identifiable direction\n(unidentifiable param combos)", fontsize=10)
    ax_dir.legend(fontsize=7, loc="best")
    ax_dir.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"FIM under different schedules — MTD (rank {decomp_mtd['effective_rank']}) vs AT50 (rank {decomp_at50['effective_rank']}) vs Periodic 56d (rank {decomp_periodic['effective_rank']}) on the same model",
        fontsize=13,
    )

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True,
        ).strip()
    except subprocess.CalledProcessError:
        sha = "uncommitted"
    date = dt.date.today().isoformat()
    base = _REPO_ROOT / "results" / "figures" / f"fig05_fim_schedule_comparison_{sha}_{date}"
    fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {base}.png and .pdf")

    # JSON summary
    summary = {
        "experiment": "fim_schedule_comparison",
        "git_sha": sha,
        "date": date,
        "param_names": PARAM_NAMES,
        "param_nominal": PARAM_NOMINAL.tolist(),
        "schedules": {
            "MTD": {
                "description": "u(t) = 1.0 for all t",
                "effective_rank": decomp_mtd["effective_rank"],
                "eigenvalues": decomp_mtd["eigenvalues"].tolist(),
                "condition_number": float(decomp_mtd["condition_number"]),
                "least_identifiable_direction": least_mtd.tolist(),
            },
            "AT50_replayed": {
                "description": "u(t) is the AT50 schedule generated at NOMINAL parameters; replayed for all FIM perturbations.",
                "n_toggles": int(np.sum(np.diff(u_at50) != 0)),
                "effective_rank": decomp_at50["effective_rank"],
                "eigenvalues": decomp_at50["eigenvalues"].tolist(),
                "condition_number": float(decomp_at50["condition_number"]),
                "least_identifiable_direction": least_at50.tolist(),
            },
            "Periodic_56d_50pct": {
                "description": "Periodic on/off schedule, period 56 days, 50% duty cycle. Multiple guaranteed toggles regardless of dynamics.",
                "effective_rank": decomp_periodic["effective_rank"],
                "eigenvalues": decomp_periodic["eigenvalues"].tolist(),
                "condition_number": float(decomp_periodic["condition_number"]),
                "least_identifiable_direction": least_periodic.tolist(),
            },
        },
        "interpretation_notes": [
            "Replayed-AT50 typically shows higher effective rank than MTD because the on-off toggling resolves drug-on phase dynamics (governed by r_S and d) separately from drug-off phase dynamics (r_R, alpha, beta).",
            "If AT50 effective rank is the same as MTD, identifiability is fundamentally model-limited rather than schedule-limited.",
            "Single-patient single-realization FIM. Multi-patient pooling deferred to Phase 3.",
        ],
    }
    summary_path = (
        _REPO_ROOT / "results" / f"fim_schedule_summary_{sha}_{date}.json"
    )
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FIM under different schedules")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (unused for deterministic FIM)")
    args = parser.parse_args()
    main()
