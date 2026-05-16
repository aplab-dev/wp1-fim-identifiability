#!/usr/bin/env bash
# Reproduce every figure cited in the WP1 methods note.
#
# Total runtime: ~60-90 minutes on an M-class CPU.
# Outputs land in results/figures/ and results/*.json with the current
# git SHA + ISO date in the filename.
#
# Prerequisites:
#   - `uv` (pip install uv) or `pip` with the deps in pyproject.toml
#   - Real-data cohorts fetched per data/raw/README.md (only if you want to
#     reproduce §6.9, §6.9.1, §6.10, §6.11)

set -e

cd "$(dirname "$0")/.."

echo "===================================================================="
echo "  Reproducing WP1 — Fisher Information Matrix Identifiability"
echo "===================================================================="

RUN() {
  local script="$1"
  local section="$2"
  echo ""
  echo "[$section] $script"
  echo "--------------------------------------------------------------------"
  uv run python "$script"
}

# §2.3 — 2-pop FIM identifiability
RUN src/experiments/04_fim_identifiability.py            "§2.3"

# §4 — Schedule invariance (2-pop)
RUN src/experiments/05_fim_schedule_comparison.py        "§4"

# §3.2 — 3-pop K-shift FIM
RUN src/experiments/08_fim_3pop_zhang.py                 "§3.2"

# §3.5 — 3-pop FIM across schedules
RUN src/experiments/12_fim_3pop_schedule_comparison.py   "§3.5"

# §5 — MCMC validation of FIM-Gaussian
RUN src/experiments/10_mcmc_synthetic_psa.py             "§5"

# §6.2 — posterior-aware policy comparison
RUN src/experiments/09_posterior_aware_policy.py         "§6.2"

# §6.4 — 1D regime scan
RUN src/experiments/13_regime_scan_policy_robustness.py  "§6.4"

# §6.6 — 2D regime scan
RUN src/experiments/15_regime_scan_2d.py                 "§6.6"

# §6.7 — synthetic cohort MCMC convergence
RUN src/experiments/14_cohort_mcmc_synthetic.py          "§6.7"

# §6.8 — PA vs PE on synthetic cohort
RUN src/experiments/16_posterior_aware_vs_point_estimate.py "§6.8"

# REAL-DATA SECTIONS — skip if data/raw/dataTanaka/ is not present.
if [ -d data/raw/dataTanaka ]; then
    # §6.9 — Bruchovsky cohort PA-vs-PE
    RUN src/experiments/19_real_cohort_pa_vs_pe.py          "§6.9"

    # §6.9.1 — Shaw cross-cohort
    RUN src/experiments/20_cross_cohort_pa_vs_pe.py         "§6.9.1"

    # §6.10 — Alpha refit to Zhang TTPs
    RUN src/experiments/21_alpha_refit_zhang_ttp.py         "§6.10"

    # §6.11 — Hierarchical Bayes on Bruchovsky
    RUN src/experiments/22_hierarchical_bruchovsky.py       "§6.11"

    # §6.11.3 — Hierarchical Bayes on Shaw (cross-cohort)
    RUN src/experiments/22_hierarchical_bruchovsky.py --cohort shaw  "§6.11.3"

    # §6.11.2 — Per-patient NUTS demo on bruchovsky_p001
    RUN src/experiments/23_nuts_real_patient.py             "§6.11.2"
else
    echo ""
    echo "NOTE: data/raw/dataTanaka not found — skipping real-data sections"
    echo "(§6.9, §6.9.1, §6.10, §6.11, §6.11.2, §6.11.3)."
    echo "Acquire per data/raw/README.md to enable."
fi

# §6.12 — Multi-modal observation channels
RUN src/experiments/24_multimodal_fim.py                  "§6.12"

# Master 4-row figure (uses outputs from above)
RUN src/experiments/17_master_figure_wp1.py               "Master figure"

echo ""
echo "===================================================================="
echo "  All figures regenerated in results/figures/"
echo "===================================================================="
ls -la results/figures/ | head -40
