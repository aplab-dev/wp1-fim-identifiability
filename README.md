# WP1: Fisher Information Matrix Identifiability of L-V Models in Adaptive Cancer Therapy

[![tests](https://github.com/aplab-dev/wp1-fim-identifiability/actions/workflows/test.yml/badge.svg)](https://github.com/aplab-dev/wp1-fim-identifiability/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Paper-CC%20BY%204.0-lightgrey.svg)](LICENSE-paper)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Companion code, data, and reproducibility artifacts for the methods note:

> **Fisher Information Matrix Identifiability of Lotka-Volterra Models in Adaptive Cancer Therapy: Rank Deficiency and Posterior-Aware Control**
>
> Aleksei Prikhodko, independent researcher · [ORCID: 0009-0003-4182-5463](https://orcid.org/0009-0003-4182-5463)
> arXiv preprint: `[ARXIV_URL — to be filled in upon submission]`

## What's in this repo

```
paper/                        ← Methods note (.md, .tex, .pdf)
src/                          ← Python library (simulators, policies, FIM, MCMC, hierarchical)
src/experiments/              ← Reproducible experiment scripts (one per major figure)
tests/                        ← pytest suite (>40 tests)
results/                      ← Pre-computed figures + JSON summaries referenced in the paper
data/raw/README.md            ← How to acquire the public Bruchovsky 2008 + Shaw 2007 cohorts
scripts/reproduce_all_figures.sh  ← One-command end-to-end reproduction
pyproject.toml                ← Python deps + tool config
```

## Quick start

```bash
git clone https://github.com/aplab-dev/wp1-fim-identifiability.git
cd wp1-fim-identifiability

# Install dependencies (uv-managed; if you don't have uv: pip install uv)
uv sync

# Acquire the public real-data cohorts (Bruchovsky 2008 + Shaw 2007)
cd data/raw
curl -sLO http://www.nicholasbruchovsky.com/dataTanaka.zip
unzip -q dataTanaka.zip
cd ../..

# Run all tests (~50s)
uv run pytest tests/

# Reproduce a single result (e.g., the FIM identifiability analysis)
uv run python src/experiments/04_fim_identifiability.py
uv run python src/experiments/08_fim_3pop_zhang.py

# Reproduce everything (~60-90 min on M-class CPU)
./scripts/reproduce_all_figures.sh
```

## Paper → script → figure mapping

See [`results/README.md`](results/README.md) for the full mapping. Quick reference:

| Paper section | Experiment script | Output |
|---|---|---|
| §2.3 FIM (2-pop) | `src/experiments/04_fim_identifiability.py` | `results/figures/fig04_*` |
| §3.2 FIM (3-pop) | `src/experiments/08_fim_3pop_zhang.py` | `results/figures/fig08_*` |
| §4 Schedule invariance | `src/experiments/05_fim_schedule_comparison.py` | `results/figures/fig05_*` |
| §5 MCMC vs FIM-Gaussian | `src/experiments/10_mcmc_synthetic_psa.py` | `results/figures/fig10_*` |
| §6.4–6 Regime scans | `src/experiments/{13,15}_*` | `results/figures/fig{13,15}_*` |
| §6.8 PA vs PE (synthetic) | `src/experiments/16_posterior_aware_vs_point_estimate.py` | `results/figures/fig16_*` |
| §6.9 Bruchovsky real cohort | `src/experiments/19_real_cohort_pa_vs_pe.py` | `results/figures/fig19_*` |
| §6.9.1 Shaw cross-cohort | `src/experiments/20_cross_cohort_pa_vs_pe.py` | `results/figures/fig20_*` |
| §6.10 α-refit (negative) | `src/experiments/21_alpha_refit_zhang_ttp.py` | `results/figures/fig21_*` |
| §6.11 Hierarchical Bayes | `src/experiments/22_hierarchical_bruchovsky.py` | `results/figures/fig22_*` |
| §6.11.2 Per-patient NUTS | `src/experiments/23_nuts_real_patient.py` | `results/figures/fig23_*` |
| §6.12 Multi-modal channels | `src/experiments/24_multimodal_fim.py` | `results/figures/fig24_*` |

## Key library entry points

```python
# 2-pop multiplicative-death model
from simulators.lv_2pop_multdeath import LV2PopMultDeath

# 3-pop K-shift (Zhang) model
from simulators.lv_3pop_kshift import LV3PopKShift
from zhang2017 import zhang_canonical_lv_params

# PSA filter
from simulators.psa_dynamics import PSAParams, psa_steady_state

# FIM computation
from identifiability import compute_fim

# Policies
from policies import MTDPolicy, AT50Policy, NoTreatmentPolicy

# Real Bruchovsky / Shaw data ingestion
from realdata import load_dataTanaka, load_shaw_et_al

# Per-patient MCMC (adaptive MH baseline)
from realdata import fit_patient_mcmc

# Per-patient NUTS (JAX-native, with hierarchical-informed priors + deterministic init)
from realdata.per_patient_hmc import fit_patient_hmc_nuts, prior_from_hierarchical_fit

# Hierarchical Bayesian fit pooling across the cohort
from realdata import hierarchical_fit, per_patient_summaries
```

## Headline findings

(from `paper/WP1_FIM_methods_note.pdf`)

1. **Structural rank deficiency.** The 2-pop multiplicative-death L-V model has FIM rank 1 of 4 under PSA-only observation. The 3-pop Zhang K-shift model has rank 3 of 6. The deficiency is *schedule-invariant* and *channel-invariant*.

2. **Rank deficiency is LOCALIZED.** The three unidentifiable directions in the 3-pop model live almost entirely in the (α(T-,T+), α(T-,TP), K_TP_drop) subspace. The three growth rates are well-identified.

3. **Multi-modal observation channels DO NOT close the rank gap.** Adding ctDNA, AR-V7, PSMA-PET, or imaging-derived TTB only improves conditioning of the identifiable subspace. Rank stays at 3 of 6.

4. **Posterior-aware control matters in clinically realistic regimes.** On the real Bruchovsky 2008 cohort (n=71), posterior-aware vs point-estimate optimal control disagrees for **37%** of patients. On the Shaw 2007 cohort (n=15), **13%**. 35% of Bruchovsky patients are "posterior-sensitive" (P(AT50 wins) ∈ [10%, 90%]).

5. **Cohort-level hierarchical pooling DOES recover identifiability.** Pooling 71 Bruchovsky patients via a non-centered hierarchical Bayesian fit shrinks the unidentifiable per-patient direction by 86-87%. Cross-cohort validation: Bruchovsky and Shaw cohorts produce population means on (α, K_TP_drop) that agree to **1-3%**. Both suggest T- competition coefficients ~50% higher than canonical Zhang.

## Data availability

The Bruchovsky 2008 IADT cohort and the Shaw et al. 2007 cohort are publicly available at:

> http://www.nicholasbruchovsky.com/dataTanaka.zip

See [`data/raw/README.md`](data/raw/README.md) for the acquisition recipe and schema.

The Zhang 2017 cohort referenced in §3 / §6.10 was not publicly available at preprint time; canonical parameter values are encoded in `src/zhang2017/`.

## Citation

If you use this code or data, please cite the arXiv preprint. A machine-readable [`CITATION.cff`](CITATION.cff) is included.

## License

- **Code** (everything under `src/`, `tests/`, `scripts/`, `pyproject.toml`): MIT — see [`LICENSE`](LICENSE).
- **Paper text** (`paper/WP1_FIM_methods_note.{md,tex,pdf}`): CC BY 4.0 — see [`LICENSE-paper`](LICENSE-paper).

## Contact

**Aleksei Prikhodko** — Independent researcher

- Website: [aplab.dev](https://www.aplab.dev/)
- Email: `aplab.official@gmail.com`
- ORCID: [0009-0003-4182-5463](https://orcid.org/0009-0003-4182-5463)
- Substack: [@alekseiprikhodko](https://substack.com/@alekseiprikhodko)
- X: [@AlekseiPrikhodk](https://x.com/AlekseiPrikhodk)

Issues and pull requests welcome. Feedback on the manuscript: please email, open an issue, or DM on X / Substack.
