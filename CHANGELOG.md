# Changelog

All notable changes to this companion-code-and-data repo for the WP1 arXiv preprint.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely.

## [Unreleased]

Reserved for post-arXiv-submission updates: arXiv ID insertion, errata, reviewer feedback.

## [0.1.0] — 2026-05-26

Initial public release coinciding with WP1 v9 arXiv submission.

### Paper

- WP1 v9 methods note (`paper/WP1_FIM_methods_note.{md,tex,pdf}`) — 17 pages, ~9000 words.
- Companion: arXiv preprint `[ARXIV_URL — to be filled in upon submission]`.

### Code

- `src/simulators/` — 2-population multdeath, 3-population K-shift, PSA filter.
- `src/policies/` — MTD, AT50, AT80, no-treatment, cohort runner.
- `src/identifiability/fim.py` — FIM computation via finite-difference sensitivities.
- `src/zhang2017/` — canonical Zhang parameter set + Stage 2.4 reproduction.
- `src/cunningham2020/` — smooth-titration optimal-control reproduction (casadi + ipopt).
- `src/realdata/` — Bruchovsky + Shaw cohort loaders, adaptive MH, NUTS via numpyro,
  custom JAX-native Heun integrator, hierarchical Bayesian fit.

### Experiments

- 17 reproducible experiment scripts (`src/experiments/03_*.py` through `24_*.py`)
  covering all WP1 figures + JSON summaries.
- `scripts/reproduce_all_figures.sh` runs them end-to-end.

### Tests

- 171 pytest tests (165 pass, 6 skip — the 6 need the Bruchovsky data file
  which users acquire per `data/raw/README.md`).
- Python 3.11+ required.
- GitHub Actions CI on push/PR for Python 3.11 + 3.12.

### Results

- All figures cited in WP1 v9 (`results/figures/fig{04,05,08,10,12,15,16,17,19,20,21,22,24}_*.{png,pdf}`).
- All JSON summaries cited in WP1 v9 (`results/{fim_summary, fim_3pop_summary, ...}.json`).

### Documentation

- `README.md` — quickstart, citation, what's in the repo.
- `CITATION.cff` — machine-readable citation metadata (CFF 1.2.0).
- `CITATION.bib` — BibTeX entry for citing the preprint.
- `data/raw/README.md` — acquisition recipe for the public dataTanaka.zip archive.
- `LICENSE` — MIT (code).
- `LICENSE-paper` — CC BY 4.0 (paper text).
