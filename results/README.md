# Results — artifact catalog

Every figure and JSON summary in this directory is paired with a paper section
and the experiment script that produced it.

## Figure → section mapping

| File | Section | Description |
|---|---|---|
| `figures/fig04_fim_identifiability_*` | §2.3 | 2-pop FIM eigenvalue spectrum |
| `figures/fig05_fim_schedule_comparison_*` | §4 | 2-pop FIM under MTD / AT50 / periodic schedules |
| `figures/fig08_fim_3pop_zhang_*` | §3.2 | 3-pop K-shift FIM at canonical Zhang θ |
| `figures/fig10_mcmc_synthetic_psa_*` | §5 | MCMC posterior vs FIM-Gaussian on synthetic data |
| `figures/fig12_fim_3pop_schedule_comparison_*` | §3.5 | 3-pop FIM across three schedules |
| `figures/fig15_regime_scan_2d_*` | §6.6 | 2D (K_TP_drop, α(T-,T+)) regime scan |
| `figures/fig16_posterior_aware_vs_point_estimate_*` | §6.8 | PA vs PE accuracy on 25-patient synthetic cohort |
| `figures/fig17_wp1_master_*` | §6 (master) | 4-row WP1 master figure ([A]-[L]) |
| `figures/fig19_real_cohort_pa_vs_pe_*` | §6.9 | Real Bruchovsky cohort PA-vs-PE breakdown |
| `figures/fig20_cross_cohort_pa_vs_pe_*` | §6.9.1 | Bruchovsky + Shaw cross-cohort comparison |
| `figures/fig21_alpha_refit_*` | §6.10 | Alpha refit to Zhang TTPs (negative result) |
| `figures/fig22_hierarchical_bruchovsky_*` | §6.11 | Hierarchical posterior on Bruchovsky (n=71) |
| `figures/fig22_hierarchical_shaw_*` | §6.11.3 | Hierarchical posterior on Shaw (n=17) |
| `figures/fig24_multimodal_fim_*` | §6.12 | Multi-modal observation channel FIM analysis |

## JSON summary → section mapping

Each JSON contains seed, settings, and the headline numbers used in the paper.

| File pattern | Section | Headline content |
|---|---|---|
| `fim_summary_*.json` | §2.3 | 2-pop FIM eigenvalues, condition number |
| `fim_3pop_summary_*.json` | §3.2 | 3-pop FIM eigenvalues, effective rank |
| `fim_schedule_summary_*.json` | §4 | Cross-schedule eigenvalue comparison |
| `fim_3pop_schedule_summary_*.json` | §3.5 | 3-pop cross-schedule comparison |
| `mcmc_synthetic_summary_*.json` | §5 | MCMC vs FIM-Gaussian std table |
| `posterior_aware_summary_*.json` | §6.2 | P(AT50 wins) at canonical Zhang |
| `regime_scan_summary_*.json` | §6.4 | 1D K_TP_drop regime scan |
| `regime_scan_2d_summary_*.json` | §6.6 | 2D (K_TP_drop, α) regime scan |
| `cohort_mcmc_summary_*.json` | §6.7 | Per-patient MH R-hat distribution |
| `decision_comparison_summary_*.json` | §6.8 | Synthetic-cohort PA-vs-PE accuracy table |
| `real_cohort_pa_vs_pe_summary_*.json` | §6.9 | Bruchovsky 37% disagreement; per-patient results |
| `cross_cohort_pa_vs_pe_summary_*.json` | §6.9.1 | Bruchovsky + Shaw side-by-side |
| `alpha_refit_summary_*.json` | §6.10 | Nelder-Mead trajectory + final α |
| `hierarchical_bruchovsky_summary_*.json` | §6.11 | Bruchovsky population posterior + shrinkage table |
| `hierarchical_shaw_summary_*.json` | §6.11.3 | Shaw cross-cohort hierarchical |
| `multimodal_fim_summary_*.json` | §6.12 | FIM rank across 9 channel combinations |
| `nuts_real_patient_p001_init_canonical_*.json` | §6.11.2 | NUTS with deterministic init — R-hat 222 |
| `nuts_real_patient_p001_hier_prior_*.json` | §6.11.2 | NUTS with hier-informed prior (negative) |
| `nuts_real_patient_p001_fim_prior_*.json` | §6.11.2 | NUTS with FIM-eigenbasis prior (negative) |
| `nuts_real_patient_p001_hier_prior_hardguard_*.json` | §6.11.2 | NUTS with hard NaN-guard (negative) |

## Filename convention

`<artifact_name>_<git-short-sha>_<YYYY-MM-DD>.{json,png,pdf}`

The git SHA and date encode when the artifact was generated. To regenerate
under a fresh checkout, the SHA portion will differ but the content should
match within seed-determined numerical tolerance.

## Reproducing

```bash
# Single artifact:
uv run python src/experiments/24_multimodal_fim.py

# All paper artifacts (~60-90 min):
./scripts/reproduce_all_figures.sh
```

See the top-level [`README.md`](../README.md) for full setup instructions.
