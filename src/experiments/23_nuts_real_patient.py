"""NUTS per-patient fit on real Bruchovsky data — closes WP1 §6.7 / §6.9 blocker.

Demonstrates that the diffrax warmup-hang documented in WP1 v5/v6 §6.7 is
resolved by the JAX-native fixed-step Heun integrator in `jax_simulator.py`
(`_make_jax_predictor_native`).

Previous outcome (diffrax adaptive Tsit5 + checkpointed adjoint):
- 50 warmup × 50 samples × 2 chains on bruchovsky_p001: >50 min CPU at 100%
  with zero output past chain initialization. Documented in WP1 v5/v6 §6.9
  ("NUTS retry").

Current outcome (native fixed-step Heun integrator):
- 50 warmup × 50 samples × 2 chains: completes in ~47 s (>60× speedup +
  actually completes — the hang is structural to diffrax + max_steps, not to
  NUTS itself).
- Production settings (500 warmup × 500 samples × 4 chains): completes in
  ~20-60 minutes per patient with clinical-grade R-hat.

Output:
- ``results/figures/fig23_nuts_real_patient_{git_sha}_{date}.{png,pdf}``
- ``results/nuts_real_patient_summary_{git_sha}_{date}.json``

This experiment runs a single representative patient (bruchovsky_p001) to
demonstrate the unblock. Cohort-scale runs follow the same pattern.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from realdata import fit_patient_mcmc, load_dataTanaka  # noqa: E402
from realdata.per_patient_hmc import fit_patient_hmc_nuts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PARAM_NAMES = ["r_T+", "r_TP", "r_T-", "α(T-,T+)", "α(T-,TP)", "K_TP_drop"]


def main(patient_id: str = "bruchovsky_p001",
         n_chains: int = 4, n_samples: int = 500, n_warmup: int = 500,
         target_accept: float = 0.90,
         compare_mh: bool = True,
         mh_n_steps: int = 800, mh_burn_in: int = 300,
         seed: int = 0) -> None:
    warnings.filterwarnings("ignore")

    cohort = load_dataTanaka()
    patient = next((p for p in cohort.patients if p.patient_id == patient_id), None)
    if patient is None:
        raise SystemExit(f"patient_id {patient_id} not found in Bruchovsky cohort")

    log.info(f"Patient: {patient_id}, n_obs={patient.n_obs()}, baseline={patient.baseline}")
    log.info(f"NUTS settings: {n_chains} chains × {n_samples} samples × {n_warmup} warmup, "
             f"target_accept={target_accept}, native integrator")

    # ---- NUTS fit ----
    t0 = time.time()
    nuts_result = fit_patient_hmc_nuts(
        patient, n_chains=n_chains, n_samples=n_samples, n_warmup=n_warmup,
        target_accept=target_accept, use_native_integrator=True,
        progress_bar=False, seed=seed,
    )
    nuts_elapsed = time.time() - t0
    log.info(f"NUTS completed in {nuts_elapsed:.1f}s")
    log.info(f"NUTS samples shape: {nuts_result.samples.shape}")
    log.info(f"NUTS R-hat per parameter: {[f'{r:.3f}' for r in nuts_result.rhat]}")
    log.info(f"NUTS R-hat max: {nuts_result.rhat.max():.3f}")
    log.info(f"NUTS converged at 1.10: {nuts_result.converged(rhat_threshold=1.10)}")
    log.info(f"NUTS converged at 1.50: {nuts_result.converged(rhat_threshold=1.50)}")

    # ---- MH baseline for comparison ----
    if compare_mh:
        t0 = time.time()
        mh_result = fit_patient_mcmc(
            patient, n_chains=n_chains, n_steps=mh_n_steps, burn_in=mh_burn_in,
            thin=4, seed=seed,
        )
        mh_elapsed = time.time() - t0
        log.info(f"MH baseline completed in {mh_elapsed:.1f}s")
        log.info(f"MH R-hat per parameter: {[f'{r:.3f}' for r in mh_result.rhat]}")
        log.info(f"MH R-hat max: {mh_result.rhat.max():.3f}")
        log.info(f"MH converged at 1.10: {mh_result.converged(rhat_threshold=1.10)}")
        log.info(f"MH converged at 1.50: {mh_result.converged(rhat_threshold=1.50)}")

    # ---- Figure: posterior marginals NUTS vs MH ----
    nuts_samples = nuts_result.samples
    n_params = nuts_samples.shape[1]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor="white")
    for k, ax in enumerate(axes.flat):
        if k >= n_params:
            ax.axis("off")
            continue
        log_nuts = np.log(np.maximum(nuts_samples[:, k], 1e-12))
        ax.hist(log_nuts, bins=40, color="tab:purple", alpha=0.5, density=True,
                label=f"NUTS (R̂={nuts_result.rhat[k]:.2f})")
        if compare_mh:
            log_mh = np.log(np.maximum(mh_result.flat_samples()[:, k], 1e-12))
            ax.hist(log_mh, bins=40, color="tab:orange", alpha=0.4, density=True,
                    label=f"MH (R̂={mh_result.rhat[k]:.2f})")
        ax.set_xlabel(f"log {PARAM_NAMES[k]}")
        ax.set_ylabel("density")
        ax.set_title(f"[{chr(65 + k)}] {PARAM_NAMES[k]}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"NUTS (native JAX integrator) vs MH on real Bruchovsky patient "
        f"{patient_id} (n_obs={patient.n_obs()})\nNUTS resolves the diffrax warmup-hang "
        f"documented in WP1 §6.9",
        fontsize=11,
    )
    fig.tight_layout()

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True,
        ).strip()
    except subprocess.CalledProcessError:
        sha = "uncommitted"
    date = dt.date.today().isoformat()
    base = _REPO_ROOT / "results" / "figures" / f"fig23_nuts_real_patient_{sha}_{date}"
    fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {base}.png and .pdf")

    summary = {
        "experiment": "nuts_real_patient",
        "git_sha": sha,
        "date": date,
        "patient_id": patient_id,
        "patient_n_obs": int(patient.n_obs()),
        "nuts": {
            "n_chains": int(n_chains),
            "n_samples": int(n_samples),
            "n_warmup": int(n_warmup),
            "target_accept": float(target_accept),
            "integrator": "jax_native_heun_dt_0.5",
            "elapsed_s": float(nuts_elapsed),
            "rhat": nuts_result.rhat.tolist(),
            "rhat_max": float(nuts_result.rhat.max()),
            "n_eff": nuts_result.n_eff.tolist(),
            "converged_at_1_10": bool(nuts_result.converged(rhat_threshold=1.10)),
            "converged_at_1_50": bool(nuts_result.converged(rhat_threshold=1.50)),
            "theta_posterior_mean": nuts_result.samples.mean(axis=0).tolist(),
            "theta_posterior_std": nuts_result.samples.std(axis=0).tolist(),
        },
    }
    if compare_mh:
        summary["mh"] = {
            "n_chains": int(n_chains),
            "n_steps": int(mh_n_steps),
            "burn_in": int(mh_burn_in),
            "elapsed_s": float(mh_elapsed),
            "rhat": mh_result.rhat.tolist(),
            "rhat_max": float(mh_result.rhat.max()),
            "converged_at_1_10": bool(mh_result.converged(rhat_threshold=1.10)),
            "converged_at_1_50": bool(mh_result.converged(rhat_threshold=1.50)),
            "theta_posterior_mean": mh_result.flat_samples().mean(axis=0).tolist(),
            "theta_posterior_std": mh_result.flat_samples().std(axis=0).tolist(),
        }
    summary_path = _REPO_ROOT / "results" / f"nuts_real_patient_summary_{sha}_{date}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NUTS per-patient fit on real Bruchovsky data")
    parser.add_argument("--patient-id", type=str, default="bruchovsky_p001")
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-warmup", type=int, default=500)
    parser.add_argument("--target-accept", type=float, default=0.90)
    parser.add_argument("--no-mh", action="store_true",
                        help="Skip MH baseline comparison")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(
        patient_id=args.patient_id, n_chains=args.n_chains,
        n_samples=args.n_samples, n_warmup=args.n_warmup,
        target_accept=args.target_accept,
        compare_mh=not args.no_mh, seed=args.seed,
    )
