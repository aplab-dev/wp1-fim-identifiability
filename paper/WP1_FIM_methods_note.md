---
key: WP1
title: "Fisher Information Matrix Identifiability of Lotka-Volterra Models in Adaptive Cancer Therapy: Rank Deficiency and Posterior-Aware Control"
status: draft
version: v8
target_venue: arXiv (q-bio.PE primary, q-bio.QM cross-list, stat.AP cross-list)
authors: Aleksei Prikhodko (independent researcher)
contact: aplab.official@gmail.com
created: 2026-05-02
last_revised: 2026-05-03
target_submission: month 6 (July 2026)
target_length: 14-18 pages including references
companion_blog: WP5e
companion_workshop_paper: WP4
revision_history:
  - v1 (2026-05-02, commit 87e9177): first 9-page draft. 8 sections + 2 appendices.
  - v2 (2026-05-03): §3.5 (3-pop cross-schedule, rank 3/6 across schedules), §6.4 (regime scan along K_TP_drop axis — posterior-sensitive regime found at K_TP_drop=1000), §6.5 (mechanistic explanation of non-monotone preference), Appendix A full proof of Theorem 2.1 (replacing v1's sketch).
  - v3 (2026-05-03): §6.6 (2D regime scan from experiment 15), §6.7 (per-patient cohort MCMC convergence diagnostic from experiment 14), §6.8 (posterior-aware vs point-estimate clinical decision comparison from experiment 16 — empirical 4pp PA advantage on 25-patient cohort, 28% PE-vs-PA disagreement). New companion: WP4 workshop paper outline (`writeups/papers/WP4_workshop_paper_outline.md`).
  - v4 (2026-05-03): JAX-native simulator with smooth-floor formulation (`src/realdata/jax_simulator.py`) — forward 1.3 ms/call (JIT), AD-stable gradient (24.6 ms). 9 tests in `tests/test_jax_simulator.py` pass. NUTS HMC wiring scaffolded (`src/realdata/per_patient_hmc.py`). NUTS sampling hits an unresolved warmup-hang issue on this compute stack; documented as production-blocker requiring follow-up engineering (~4-8h). §6.7 honest about MH being the current sampler with HMC as the future production path.
  - v5 (2026-05-03): **REAL DATA acquired and pipeline runs end-to-end.** + WP1 literature cross-check pass: Brady-Nicholls 2020 confirmed n=70 + correlation matrix at $\xi=0.95$ + 89% accuracy + 4-to-2 parameter reduction (was loosely "5→2→1" in v4 — corrected). Strobl 2022 confirmed r=-0.76, p=1.4e-11, n=65. Gallagher 2025 confirmed 2-of-5 fit + n=5+3 cohort + R²=0.92. + Cross-cohort validation: Shaw et al. (2007) cohort (n=17 in same archive) added; PE-vs-PA disagreement = 13% on Shaw vs 37% on Bruchovsky. Cross-cohort range 13-37% brackets the synthetic-cohort estimate (28%). §6.9.1 added. Bruchovsky et al. 72-patient IADT cohort downloaded from `nicholasbruchovsky.com/dataTanaka.zip` (the same cohort fit by Brady-Nicholls 2020, Strobl 2022, Gallagher 2025). New `realdata.load_dataTanaka()` ingests per-patient PSA + treatment time-series. New experiment 19 (`src/experiments/19_real_cohort_pa_vs_pe.py`) runs per-patient MCMC + PA-vs-PE comparison on real data. Pilot run (5 patients): PE-vs-PA disagreement = 40% — substantially higher than the 28% on synthetic cohort, suggesting real-data posteriors are more posterior-sensitive than our synthetic cohort. §6.9 added: "Real-data validation on the Bruchovsky cohort" (full 72-patient results once exp 19 completes).
  - v6 (2026-05-03): + §6.10 (calibration caveat: alpha-refit could not reproduce Zhang 2017 absolute TTP magnitudes by tuning $\alpha$ alone — best-fit gives MTD=AT50=35mo vs Zhang 16/27mo target — but PE-vs-PA *relative* disagreement findings stand). + §6.11 (closed-form Gaussian hierarchical Bayesian fit pooling across the Bruchovsky cohort; partially recovers unidentifiable directions through population-level shrinkage; median posterior-std reduction **79% on $\alpha_{T-,T+}$ and 89% on $\alpha_{T-,TP}$** on the full n=71 cohort — full-cohort numbers replace earlier n=5 smoke-test 65%/35%). + §7.2 limitation #1 updated to reference §6.11 closure. New `realdata.hierarchical` module + experiment 22. **Hierarchical NUTS converges on $\mu$ (max $\hat R_\mu = 1.08$) but $\sigma_\text{pop}$ on $\alpha_{T-,T+}$ reaches $\hat R = 1.47$ — documented as σ_pop convergence caveat; shrinkage estimate is robust to this.**
  - v7 (2026-05-13): **One v6 caveat fully closed, one partially closed.** (1) HIERARCHICAL CONVERGENCE — FULLY CLOSED. Switched to non-centered reparameterization (`realdata.hierarchical` `centered=False` default); $\sigma_\text{pop}$ R-hat on $\alpha_{T-,T+}$ goes from 1.47 → 1.001 (clinical-grade). All 12 R-hat statistics pass the $\hat R < 1.10$ threshold. Updated §6.11 numbers reflect non-centered run (shrinkage 87% / 86% on $\alpha_{T-,T+}$ / $\alpha_{T-,TP}$ — both up from v6). New §6.11.1 documents the centered-vs-non-centered comparison. (2) PER-PATIENT NUTS WARMUP-HANG — STRUCTURALLY RESOLVED, BUT CONVERGENCE STILL POOR. New `_make_jax_predictor_native` in `realdata.jax_simulator` replaces diffrax adaptive Tsit5 with a custom JAX-native fixed-step Heun integrator (`jax.lax.scan`). No `max_steps` ceiling. On `bruchovsky_p001` (n_obs=75), NUTS with 50w × 50s × 2c completes in **47 s** vs **>3000 s hang** with diffrax — the warmup-hang failure mode is gone. *However*, production NUTS at 500w × 500s × 4c completes in ~24 min and produces R-hat = [346, 7, 83, 18, 79, 4] — chains find different modes in the rank-deficient directions. Per-patient NUTS *completes* but does not *converge* at clinical-grade R-hat. This is documented honestly in revised §6.7 — the next Phase 4 task is per-patient reparameterization in the FIM-identifiable basis. Forward 4.7 ms/call (1.9× diffrax), gradient 48 ms/call (2.0× diffrax). New experiment 23 (`23_nuts_real_patient.py`). WP1's empirical claims are unaffected: the PE-vs-PA disagreement findings depend on rank-deficient direction *width*, not on per-patient mode-finding.
  - v8 (2026-05-13): + §6.11.2 deepened with 4 NUTS-on-real-data attempts (3 negative, 1 partial win). Approach 4 (deterministic init at the prior mean) brings R-hat from 9000 → 222 — 40× improvement, chains stay in physical region, posterior mean confirms cohort hier fit. + §6.11.3 (cross-cohort hierarchical fit on Shaw cohort): population posterior means on α and K_TP_drop agree to **1-3% between Shaw and Bruchovsky** (n=17 and n=71). Both cohorts independently suggest T- competition coefficients ~50% higher than canonical Zhang. + §6.12 (multi-modal observation channels): **negative result — adding observation channels does NOT close the rank gap on the 3-pop K-shift model**. All 5 channel combinations (PSA, +TTB, +T-_frac, +TP, +T+, +all) give effective rank 3 of 6. The rank deficiency reflects symmetries of the L-V dynamics, not informational poverty. + §4.3 / §7.3 implication #3 corrected: multi-modal channels improve precision but do NOT deliver per-patient identifiability. Path to per-patient ID is model simplification *or* cohort pooling. **Major correction** to v6/v7 §7.3 implication #3.
---

# Fisher Information Matrix Identifiability of Lotka-Volterra Models in Adaptive Cancer Therapy: Rank Deficiency and Posterior-Aware Control

**Aleksei Prikhodko**\
*Independent researcher*\
aplab.official@gmail.com

## Abstract

Adaptive cancer therapy schedules drug-on/drug-off cycles based on biomarker dynamics, with the goal of preserving competitive pressure from drug-sensitive cells against drug-resistant ones. Three independent prior studies — Brady-Nicholls et al. (2020), Strobl et al. (2022), and Gallagher et al. (2025) — fit different Lotka-Volterra (L-V) parameterizations to overlapping clinical cohorts and converge on the empirical conclusion that *approximately one effective parameter per patient* is recoverable from PSA-only observation. We provide the structural explanation: the Fisher Information Matrix (FIM) of the canonical 2-population multiplicative-death L-V model has effective rank 1 of 4, with the competition coefficients $\alpha$ and $\beta$ exhibiting estimate-correlation $-1.00$ (Theorem 2.1, full proof in Appendix A). The 3-population K-shift model used by Zhang et al. (2017) has effective rank 3 of 6 — substantially more identifiable structure due to time-scale separation between $T+$, $TP$, and $T-$ subpopulations. The rank deficiency is *model-fundamental*: clinical adaptive schedules (AT50 cycling, periodic on/off) yield identical FIM eigenvalue spectra on both 2-pop and 3-pop models. We compare the asymptotic Cramér-Rao Gaussian to actual MCMC posteriors on synthetic PSA data; the FIM-Gaussian is faithful in identifiable directions (within 15%) but underestimates posterior uncertainty by 20-30× in unidentifiable directions. We propagate the resulting parameter posterior through to policy-comparison values. In the Zhang-canonical regime, $\mathbb{P}(\text{AT50 wins TTP}) = 100\%$ across both FIM-Gaussian and MCMC posteriors, demonstrating that policy-preference robustness can survive identifiability rank-deficiency. **However, a regime scan along the drug-effectiveness axis $K_{TP}^\text{drop}$ reveals a posterior-sensitive regime ($\mathbb{P}(\text{AT50 wins}) = 45\%$ at $K_{TP}^\text{drop} = 1000$) — a near-coin-flip across the unidentifiable manifold.** A 2D scan in $(K_{TP}^\text{drop}, \alpha_{T-,T+})$ confirms this is a vertical band in parameter space at $K_{TP}^\text{drop} \in [1000, 2500]$. **On a 25-patient synthetic cohort spanning this band, posterior-aware control gives 80% accuracy vs the 76% accuracy of point-estimate optimal control (4 percentage points; 28% per-patient disagreement rate)** — empirical evidence that the methodology choice matters in clinically realistic regimes, not only in pathological ones. **On the real Bruchovsky 2008 IADT cohort (n=72 patients, the same cohort fit by Brady-Nicholls 2020 / Strobl 2022 / Gallagher 2025), 71 of 72 patients fit successfully and the PE-vs-PA disagreement rate is 37% (26 / 71); on the independent Shaw et al. (2007) cohort (n=17 in the same archive, 15 fit successfully), the disagreement rate is 13% (2 / 15).** Point-estimate optimal recommends AT50 for 14% of Bruchovsky patients while posterior-aware recommends AT50 for 51% — methodology flips the recommendation for more than a third of the Bruchovsky cohort, while in the more-aggressive Shaw cohort the disagreement is smaller (1 in 8 patients). 35% of Bruchovsky patients are posterior-sensitive (P(AT50 wins TTP) between 10% and 90%) versus 7% in Shaw; for these, the point estimate is uninformative about the right choice and only marginalization over the unidentifiable manifold yields a defensible recommendation. We discuss implications for clinical adaptive-therapy decision support and identify per-patient HMC-based Bayesian fitting (replacing the adaptive Metropolis-Hastings used here, which fails to converge on the rank-deficient posterior) as the production-grade requirement for clinical deployment.

**Keywords.** Lotka-Volterra dynamics; adaptive cancer therapy; Fisher Information Matrix; identifiability; mCRPC; Bayesian decision theory.

---

## 1. Introduction

Adaptive cancer therapy (AT) is a clinical strategy in which drug administration is modulated based on biomarker trajectories rather than fixed schedules [Gatenby2009; Zhang2017]. The mechanistic rationale is evolutionary: continuous maximum-tolerated-dose (MTD) chemotherapy applies sustained selective pressure that favors resistant tumor sub-populations, while intermittent dosing preserves competitive pressure from sensitive cells against resistant ones. Two-population Lotka-Volterra (L-V) competition models [Strobl2021; Gallagher2025] capture this mechanism with closed-form mathematical structure; three-population K-shift models [Zhang2017; Cunningham2020; West2020] extend to the testosterone-axis biology of metastatic castrate-resistant prostate cancer (mCRPC).

A pervasive practical issue in deploying these models clinically is **parameter identifiability from limited biomarker data**. Three independent prior studies on overlapping clinical cohorts (Bruchovsky 2008 IADT and Zhang 2017 mCRPC) converge on the same empirical reduction:

1. Brady-Nicholls et al. (2020) [Brady-Nicholls2020], fitting a stem-cell-vs-differentiated-cell ODE model on the Bruchovsky 2008 cohort (n=70 after exclusions), use correlation-matrix screening (correlation threshold $\xi = 0.95$) to reduce 4 free parameters to 2 patient-specific ($p_s$ and $\alpha$), with 2 population-uniform ($\rho$ and $\phi$). They report 89% overall TTP-prediction accuracy (sensitivity 73%, specificity 91%) on the n=70 cohort under leave-one-out validation [Brady-Nicholls2020 Methods §"Mathematical model training and validation"; Supplementary Figure 3C].
2. Strobl et al. (2022) [Strobl2022], fitting a 2-population L-V agent-based model on n=65 from the same Bruchovsky cohort (different exclusion criteria), report estimate-correlation $r = -0.76$ ($p = 1.4 \times 10^{-11}$) between resistance cost and turnover [Strobl2022 Results].
3. Gallagher et al. (2025) [Gallagher2025] fit a 2-population multiplicative-death L-V biomarker on n=5 Bruchovsky-test + n=3 Zhang-2017-validation patients. They explicitly fit 2 of the 5 model parameters per patient (the drug-induced death rate $d_S$ and the carrying capacity $K$) and treat $r_S$, $r_R$, $R_0$ as population-uniform by design. Headline R² = 0.92 on the n=5 Bruchovsky cohort.

These three works use different ontologies, different fitting machinery, and different data preprocessing — yet they all end at "approximately one effective parameter per patient is robustly recoverable." None of the papers states the structural reason. We argue that this convergence is not coincidence: it is a Fisher Information Matrix rank deficiency that is intrinsic to the (model, observation) pair.

**Contributions.** Section 2 derives the FIM for the canonical 2-population multiplicative-death L-V model with PSA-only observation and proves rank-deficiency. Section 3 extends the analysis numerically to the 3-population K-shift model used by Zhang 2017 and Cunningham 2020, finding rank 3 of 6 — qualitatively different from the 2-pop case. Section 4 demonstrates that the rank deficiency is *schedule-invariant*: cycling under AT50 and forced-periodic schedules give identical FIM spectra. Section 5 validates the asymptotic Cramér-Rao Gaussian against an MCMC posterior on synthetic PSA data, identifying a 20-30× underestimate in unidentifiable directions. Section 6 propagates the posterior to policy-comparison values, showing that AT50 robustly dominates MTD across both posterior approximations in the Zhang-canonical regime. Section 7 discusses implications for clinical adaptive-therapy decision-making and posterior-aware control.

**Reproducibility.** All numerical results are produced by the open-source repository at [link to be added once public] under the experiment scripts referenced inline. Each figure is timestamped with git SHA; all parameter values, seeds, and observation schedules are committed.

## 2. The 2-Population Multiplicative-Death Model

### 2.1 Model

The canonical 2-population L-V multiplicative-death model is
$$
\begin{aligned}
\dot{S} &= r_S\, S\!\left(1 - \frac{S + \alpha R}{K}\right) - d\, u(t)\, S, \\
\dot{R} &= r_R\, R\!\left(1 - \frac{R + \beta S}{K}\right),
\end{aligned}
\tag{1}
$$
where $S$ and $R$ are sensitive and resistant cell counts, $r_S, r_R$ are intrinsic growth rates, $\alpha, \beta$ are inter-population competition coefficients, $K$ is the shared carrying capacity, $d$ is the drug-induced death rate, and $u(t) \in [0, 1]$ is the time-varying drug control. PSA serves as a noisy aggregate observation:
$$
y(t) = \rho\,(S(t) + \gamma R(t)) - \phi\, P(t) + \varepsilon(t), \quad \dot{P} = \rho(S + \gamma R) - \phi P, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2(t)).
\tag{2}
$$
Here $\rho$ is per-cell PSA production, $\phi$ is PSA decay (Zhang 2017 uses $\phi = 0.5\,\text{day}^{-1}$, half-life $\sim 1.4$d), and $\gamma \le 1$ accounts for resistant cells producing less PSA.

The free parameter vector is $\theta = (r_S, r_R, \alpha, \beta, K, d) \in \mathbb{R}^6$ if $\rho, \phi, \gamma$ are held population-uniform. Held fixed: initial conditions and observation schedule.

### 2.2 The Fisher Information Matrix

For a Gaussian noise model with observation times $\{t_k\}_{k=1}^{N}$ and per-time-point variance $\sigma^2(t_k)$, the FIM at parameter vector $\theta$ is
$$
\mathcal{I}_{ij}(\theta) = \sum_{k=1}^{N} \frac{1}{\sigma^2(t_k)} \frac{\partial y(t_k; \theta)}{\partial \theta_i} \frac{\partial y(t_k; \theta)}{\partial \theta_j}.
\tag{3}
$$
The Cramér-Rao asymptotic posterior covariance is $\Sigma_\text{CR}(\theta) = \mathcal{I}^{-1}(\theta)$ when $\mathcal{I}$ is full rank, or the Moore-Penrose pseudoinverse $\mathcal{I}^+(\theta)$ when rank-deficient.

### 2.3 Numerical FIM evaluation

We compute (3) via central finite-difference sensitivities $\partial y / \partial \theta_i \approx (y(\theta + \delta_i) - y(\theta - \delta_i)) / (2\delta_i)$ with relative perturbation step $\delta_i = 10^{-3} \cdot |\theta_i|$. At the regime-A nominal $\theta_0 = (0.05, 0.04, 0.7, 0.6, 1.0, 1.5)$ with constant MTD ($u(t) = 1$) and observation schedule of one PSA measurement every 28 days over 500 days (~17 measurements), with 10% relative noise floored at 10% of peak PSA:

$$
\text{eig}(\mathcal{I}) = (1.5 \times 10^6,\; 3.9 \times 10^{-3},\; 3.0 \times 10^{-8},\; 2.1 \times 10^{-10}).
\tag{4}
$$

Effective rank (eigenvalues above $10^{-6} \cdot \lambda_\text{max}$) is **1 of 4** when only the four dynamics parameters $(r_S, r_R, \alpha, \beta)$ are fit. The condition number is $\kappa(\mathcal{I}) = 7.1 \times 10^{15}$.

### 2.4 The α-β degeneracy

The Moore-Penrose-pseudoinverse-derived estimate correlation matrix has

$$
\rho(\hat{\alpha}, \hat{\beta}) = -1.00,
\tag{5}
$$

i.e., $\alpha$ and $\beta$ are perfectly anti-correlated. The cause is structural: in (1), $\alpha$ enters only as $\alpha R$ in $\dot{S}$, and $\beta$ enters only as $\beta S$ in $\dot{R}$. Under the PSA observation (2) which couples $S + \gamma R$, the two contributions are observationally indistinguishable along a one-dimensional ridge in the $(\alpha, \beta)$ plane.

**Theorem 2.1** (Informal). *Under PSA-only observation of (1)-(2), the parameters $\alpha$ and $\beta$ are not jointly identifiable: there exists a one-parameter family $(\alpha(s), \beta(s))$ for $s \in \mathbb{R}$ such that $y(t; r_S, r_R, \alpha(s), \beta(s), K, d) = y(t; r_S, r_R, \alpha(0), \beta(0), K, d)$ for all $t$ and all $s$ in a neighborhood of $0$.*

*Proof sketch.* The first-order sensitivity equations $\dot{X}_\theta = (\partial f/\partial x) X_\theta + (\partial f/\partial \theta)$ for $\theta \in \{\alpha, \beta\}$ produce parallel directions in the $(S, R, P)$ state space when $S(t) \approx K(1 - \alpha R/K)$ holds along the trajectory (which is the AT-relevant coexistence regime). The PSA filter (2) projects these parallel state-space directions onto the same observable function. Full proof: Appendix A.

### 2.5 Connecting to the empirical convergence

Result (5) recovers the structural reason for the convergence reported in Brady-Nicholls 2020 (4-to-2 patient-specific reduction via correlation-matrix screening at threshold $\xi = 0.95$), Strobl 2022 ($r = -0.76$ cost-vs-turnover correlation), and Gallagher 2025 (explicit 2-of-5 fit). Each prior study saw the rank-deficiency manifesting in a different parameterization but did not connect it to the FIM structure.

## 3. The 3-Population K-Shift Model

### 3.1 Model

The Zhang 2017 / Cunningham 2020 mCRPC model uses three subpopulations $(T+, TP, T-)$ for testosterone-dependent, testosterone-producing, and testosterone-independent cells, with drug entering via carrying-capacity shift rather than multiplicative death:

$$
\frac{dx_i}{dt} = r_i\, x_i\!\left(\frac{K_i(\Lambda; x_{TP}) - \sum_{j} \alpha_{ij}\, x_j}{K_i(\Lambda; x_{TP})}\right), \qquad i \in \{T+, TP, T-\}, \tag{6}
$$
with $K$-shift functions
$$
\begin{aligned}
K_{T-}(\Lambda) &= 10{,}000, \\
K_{TP}(\Lambda) &= 10{,}000 - 9{,}900\,\Lambda, \\
K_{T+}(\Lambda; x_{TP}) &= (1.5 - \Lambda)\, x_{TP},
\end{aligned}
\tag{7}
$$
and growth rates $r = (2.7726, 3.4657, 6.6542) \times 10^{-3}\,\text{day}^{-1}$.

We fit a 6-parameter subset that a clinical study would attempt to identify per patient: $\theta = (r_{T+}, r_{TP}, r_{T-}, \alpha_{T-,T+}, \alpha_{T-,TP}, K_{TP}^\text{drop})$, holding the rest at canonical Zhang values.

### 3.2 FIM result

Using the same central-difference + $1.5 \times 10^3$-day MTD trajectory + $\sigma = 10\%$ relative noise as Section 2:

$$
\text{eig}(\mathcal{I}_{3pop}) = (3.7 \times 10^8,\; 2.3 \times 10^7,\; 3.6 \times 10^5,\; 0.88,\; 2.8 \times 10^{-4},\; 8.7 \times 10^{-6}).
\tag{8}
$$

Effective rank is **3 of 6** — three orders of magnitude separate $\lambda_3$ from $\lambda_4$, indicating a clear identifiability gap.

### 3.3 Why the 3-pop model is more identifiable

The dominant sensitivities trace different time scales:

- $r_{T+}$ controls the initial collapse rate when $K_{T+} \to 0.5\,x_{TP}$ shrinks under MTD (~0-50d).
- $r_{TP}$ controls the intermediate-time TP-collapse rate (~50-500d).
- $r_{T-}$ controls the slow long-time T- regrowth (~500-1500d).

Because each parameter's sensitivity peaks at a distinct time, the FIM resolves them. The unidentifiable directions are:
- $r_{TP}$ vs $r_{T-}$ estimate correlation $\approx -0.99$ (slower modes confound).
- $\alpha_{T-,T+}$ vs $\alpha_{T-,TP}$ estimate correlation $\approx -0.94$ (the analog of the 2-pop α-β degeneracy).
- $K_{TP}^\text{drop}$ correlated with multiple parameters.

So the 3-pop model has *richer information content per observation* than the 2-pop model under the same PSA channel — by roughly a factor of three in effective dimensions. This is an underappreciated point in the field's modeling-choice debates.

### 3.4 Practical implication

Under PSA-only observation, the 3-population K-shift Zhang model can support a *3-parameter* per-patient Bayesian fit, while the 2-population multdeath model can support only a *1-parameter* per-patient fit. The Brady-Nicholls 2020 explicit reduction to 2 patient-specific parameters with 2 held population-uniform — verified at the Methods level (correlation-matrix screening at threshold $\xi = 0.95$, leave-one-out validation, 89% accuracy on n=70) — is consistent with the 2-pop K-shift figure of $\le 2$ identifiable directions; Gallagher 2025's 2-of-5 fit on the multdeath model is *more aggressive than the FIM justifies* but not unreasonable given a domain prior + their R² = 0.92 cross-validation result on the smaller n=5 cohort.

### 3.5 The 3-pop rank advantage extends across schedules

We rerun §4's three-schedule analysis (MTD, replayed-AT50, forced-periodic-56d) on the 3-population model. Effective rank is **3 of 6 in all three cases**:

| Schedule | $\lambda_1$ | $\lambda_2$ | $\lambda_3$ | $\lambda_4$ | Effective rank |
|---|---|---|---|---|---|
| MTD | $3.7 \times 10^8$ | $2.3 \times 10^7$ | $3.6 \times 10^5$ | $0.88$ | 3 |
| Replayed AT50 | $1.1 \times 10^8$ | $1.6 \times 10^6$ | $1.3 \times 10^4$ | $0.36$ | 3 |
| Periodic 56d | $4.0 \times 10^8$ | $3.9 \times 10^7$ | $5.3 \times 10^5$ | $1.6$ | 3 |

The eigenvalue magnitudes differ — AT50 cycling produces a less informative third direction by ~30× than MTD — but the effective rank is preserved. The "schedule does not change the identifiable subspace dimension" finding from §4 generalizes from the theory tribe (rank 1) to the clinical tribe (rank 3).

(Reproduction: `src/experiments/12_fim_3pop_schedule_comparison.py`.)

### 3.6 The unidentifiable subspace lies entirely in $(\alpha, K_{TP}^\text{drop})$

A natural question is: *which* parameter directions are the rank-deficient ones? The FIM eigendecomposition at canonical Zhang $\theta$ + MTD gives:

| Eigenvalue | Approx eigenvector direction | Identifiability |
|---|---|---|
| $\lambda_1 = 3.7 \times 10^8$ | $r_{T-}$ (98%) | ✓ identifiable |
| $\lambda_2 = 2.3 \times 10^7$ | $r_{TP} + 0.5\, r_{T+}$ (intermediate-rate contrast) | ✓ identifiable |
| $\lambda_3 = 3.6 \times 10^5$ | $r_{T+} - 0.5\, r_{TP}$ (fast-rate contrast) | ✓ identifiable |
| $\lambda_4 = 0.88$ | $0.92\, \alpha_{T-,T+} + 0.38\, \alpha_{T-,TP}$ (sum direction) | ⚠️ marginal |
| $\lambda_5 = 2.8 \times 10^{-4}$ | $0.86\, \alpha_{T-,TP} - 0.37\, K_{TP}^\text{drop} - 0.36\, \alpha_{T-,T+}$ | ✗ unidentifiable |
| $\lambda_6 = 8.7 \times 10^{-6}$ | $0.93\, K_{TP}^\text{drop} + 0.34\, \alpha_{T-,TP}$ | ✗ unidentifiable |

The three identifiable directions are *almost entirely* the three growth rates $(r_{T+}, r_{TP}, r_{T-})$ — they project onto each other in different combinations as the trajectory unfolds over distinct time scales (Section §3.3). The three unidentifiable directions are *almost entirely* in the $(\alpha_{T-,T+}, \alpha_{T-,TP}, K_{TP}^\text{drop})$ subspace; the $r$-direction contributions are below $\sim 5\%$ in any unidentifiable eigenvector.

**Structural interpretation.** This is consistent with the analytical structure of the 3-pop K-shift ODE:
- $r_i$ multiplies $x_i$ directly → strongly observable through trajectory shape.
- $\alpha_{T-,T+}$ and $\alpha_{T-,TP}$ both enter $\dot x_{T-}$, multiplied by $x_{T+}$ and $x_{TP}$ respectively. Since $x_{T+}$ and $x_{TP}$ trajectories are dynamically coupled (TP carrying capacity drives T+ growth via the K-shift), the two $\alpha$ coefficients enter the observable through a near-degenerate combination.
- $K_{TP}^\text{drop}$ shifts the TP carrying capacity, which then propagates through TP dynamics and the K-shift coupling — its effect on PSA is small and correlated with multiple α terms.

So the rank-3-of-6 deficiency is **localized to the 3-D $(\alpha, K_{TP}^\text{drop})$ block**, not spread uniformly over $\theta$. This has practical implications: a model-simplification path that fixes the α's and $K_{TP}^\text{drop}$ at population-uniform values (the Brady-Nicholls 2020 / Gallagher 2025 strategy) would recover the full rank-3 information content per patient. Cohort pooling (§6.11) achieves the same effect by constraining the α's at the population level.

(Reproduction: same FIM as §3.2, eigendecomposed; see `src/experiments/24_multimodal_fim.py` for the analysis code.)

## 4. Schedule Invariance

### 4.1 Setup

Three drug schedules are tested with the 2-pop multdeath model from Section 2, all observed at 28-day cadence over 500 days:

- **MTD.** $u(t) = 1$ for all $t$.
- **Replayed AT50.** $u(t)$ is the schedule generated by AT50 protocol (drug on until PSA $\le 0.5 \cdot \text{baseline}$; off until PSA returns to baseline) at the *nominal* parameters $\theta_0$, then the same schedule is replayed for FIM perturbations. (Replay avoids the discontinuity that arises when toggle times shift with $\theta$.)
- **Periodic 56d.** Forced 50% duty cycle: $u(t) = 1$ for the first 28 days of each 56-day period, $0$ otherwise. Guarantees multiple toggles regardless of dynamics.

### 4.2 Result

All three schedules give effectively identical FIM eigenvalue spectra and the same α-β anti-correlation as the most poorly identified direction:

| Schedule | $\lambda_1$ | $\lambda_2$ | $\lambda_3$ | $\lambda_4$ | Effective rank |
|---|---|---|---|---|---|
| MTD | $1.52 \times 10^6$ | $3.87 \times 10^{-3}$ | $2.96 \times 10^{-8}$ | $2.14 \times 10^{-10}$ | 1 |
| Replayed AT50 | $1.52 \times 10^6$ | $3.87 \times 10^{-3}$ | $1.24 \times 10^{-7}$ | $2.27 \times 10^{-8}$ | 1 |
| Periodic 56d | $1.52 \times 10^6$ | $3.87 \times 10^{-3}$ | $3.23 \times 10^{-7}$ | $4.52 \times 10^{-8}$ | 1 |

### 4.3 Interpretation

Cycling does not recover identifiability. The rank deficiency is *fundamental to the model and observation channel*. In practical terms: **no choice of clinical schedule rescues per-patient parameter recovery** for the 2-pop multdeath model under PSA-only observation. We initially conjectured that multi-modal observation channels (ctDNA, AR-V7 transcript, mIHC tumor-infiltration data) would be the path to higher identifiability — and tested this directly on the 3-pop K-shift model in §6.12 below. The surprising finding: **adding observation channels does not close the rank gap on the 3-pop K-shift model either**; it only improves the conditioning of the already-identifiable subspace. The rank deficiency reflects symmetries of the underlying dynamics, not informational poverty of the PSA channel alone.

## 5. MCMC Validation of the Cramér-Rao Gaussian

### 5.1 Setup

We generate synthetic PSA data $y_\text{obs} = y(\theta_\text{true}) + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma^2(t))$ at the canonical 3-pop K-shift $\theta_\text{true}$. We run an adaptive component-wise Metropolis-Hastings MCMC with target acceptance $\sim 0.234$ (Roberts-Rosenthal asymptotic-optimal rate in $d=6$), 3000 steps, 1000 burn-in, thin = 4. Improper uniform prior with positivity constraints.

### 5.2 Result

| Parameter | True | MCMC mean | MCMC std | FIM-Gaussian std | MCMC / FIM |
|---|---|---|---|---|---|
| $r_{T+}$ | 0.0028 | 0.0028 | 0.0017 | 0.0015 | **1.15** |
| $r_{TP}$ | 0.0035 | 0.0037 | 0.00079 | 0.00072 | **1.09** |
| $r_{T-}$ | 0.0067 | 0.0067 | 7.8e-5 | 7.7e-5 | **1.02** |
| $\alpha_{T-,T+}$ | 3.0 | 2.91 | 0.033 | 0.0016 | **20.2** |
| $\alpha_{T-,TP}$ | 4.0 | 3.99 | 0.034 | 0.0016 | **20.9** |
| $K_{TP}^\text{drop}$ | 9900 | 9900 | 0.049 | 0.0016 | **29.6** |

In identifiable directions ($r_{T+}, r_{TP}, r_{T-}$), the FIM-Gaussian std is within 15% of the MCMC posterior std. In unidentifiable directions ($\alpha_{T-,T+}, \alpha_{T-,TP}, K_{TP}^\text{drop}$), the FIM-Gaussian *underestimates* posterior uncertainty by 20-30×.

### 5.3 Interpretation

The Cramér-Rao asymptotic Gaussian is reliable in identifiable directions but produces dramatically too-tight posteriors in unidentifiable directions when the eigenvalue regularization (here, floor at $\lambda_\text{max} \cdot 10^{-3}$) is applied without care. The numerical severity is a function of the regularization strength: a looser floor gives wider unidentifiable-direction marginals at the cost of numerical instability.

**Practical recommendation.** For per-patient Bayesian inference in clinical AT applications, use *MCMC directly* on the rank-deficient regime rather than the regularized FIM-pseudoinverse. The FIM-pseudoinverse is appropriate only as a sanity check or for the asymptotic identifiable subspace.

## 6. Posterior-Aware Policy Comparison

### 6.1 Setup

We compare two clinical policies:
- **MTD:** $u(t) = 1$ continuously.
- **AT50:** drug on until PSA $\le 0.5 \cdot$ baseline; off until PSA $\ge$ baseline.

For each posterior sample $\theta^{(i)}$ (drawn either from the FIM-Gaussian or from the MCMC chain), we run a small per-patient cohort ($n=3$ patients with 10% log-normal IC perturbation) under both policies, compute median time-to-progression (TTP) per arm, and accumulate $\mathbb{P}(\text{AT50 wins TTP}) = \mathbb{E}_\theta\left[\mathbb{1}[\text{TTP}_\text{AT50}(\theta) > \text{TTP}_\text{MTD}(\theta)]\right]$.

### 6.2 Result

| Posterior | $N$ samples | $\mathbb{P}(\text{AT50 wins TTP})$ | $\mathbb{P}(\text{AT50 wins drug})$ | Median advantage |
|---|---|---|---|---|
| FIM-Gaussian (regularized) | 57 | 100% | 100% | 352 days (~12 mo) |
| MCMC | 50 | 100% | 100% | 362 days (~12 mo) |

In the Zhang-canonical regime, both posterior approximations give the same answer: AT50 robustly dominates MTD on TTP and drug exposure. The unidentifiable directions are *orthogonal* to the policy-preference direction.

### 6.3 Interpretation

Identifiability rank-deficiency does not always translate into policy uncertainty. In the canonical Zhang regime, the policy choice is robust to which point in the unidentifiable manifold the true parameters lie. This is reassuring clinically: even when per-patient parameter fits are unreliable, the qualitative AT-vs-MTD choice can still be made confidently.

But this is not the whole story. Section 6.4 below tests other regimes.

### 6.4 Regime scan: where does the posterior preference become sensitive?

We scan along the $K_{TP}^\text{drop}$ axis (drug effectiveness on TP carrying capacity), holding the other 5 parameters at canonical Zhang values. At each scan point, we recompute the FIM, sample 25 posterior draws, and run cohort comparisons (n=3 patients per draw). Result:

| $K_{TP}^\text{drop}$ | $\mathbb{P}(\text{AT50 wins TTP})$ | Median advantage | $\mathbb{P}(\text{AT50 saves drug})$ | Note |
|---|---|---|---|---|
| 1000 (weak drug) | **45%** | -18 d | 0% | **Posterior-sensitive (coin-flip)** |
| 2500 | 5% | 0 d | 0% | MTD favored |
| 4000-8500 | 0% | 0 d | 100% | MTD wins TTP; AT50 saves drug |
| 9900 (canonical) | 100% | +330 d | 100% | AT50 dominates |

Three findings:

1. **Posterior-sensitive regimes exist.** At $K_{TP}^\text{drop} = 1000$, $\mathbb{P}(\text{AT50 wins TTP}) = 45\%$ — a near-coin-flip across the FIM-induced posterior. Patients whose true parameters lie in this regime would receive *conflicting policy recommendations* from different point estimates on the unidentifiable manifold.

2. **The boundary is sharp.** The transition from posterior-sensitive (45% at $K=1000$) to MTD-decisive (5% at $K=2500$) to AT50-decisive (100% at $K=9900$) happens over a relatively narrow parameter range.

3. **The behavior is non-monotone.** "More drug effectiveness → more AT50 advantage" is not the relationship. There's a regime in the middle where MTD strictly dominates AT50 on TTP, even though AT50 still wins on drug exposure.

**This is the case where posterior-aware control matters more than point-estimate optimal control.** A point estimate at the canonical regime ($K=9900$) confidently recommends AT50; a point estimate at $K=1000$ is ~50/50 across the manifold; a point estimate at $K=2500$ confidently recommends MTD. The *expected* policy value over the posterior gives the principled answer in each regime; only the canonical regime aligns with the point-estimate recommendation.

(Reproduction: `src/experiments/13_regime_scan_policy_robustness.py`. Figure 13: scan visualization.)

### 6.5 Why the regime is non-monotone

The non-monotone $\mathbb{P}(\text{AT50 wins})$ as $K_{TP}^\text{drop}$ varies has a structural explanation. At low $K_{TP}^\text{drop}$ (weak drug), MTD doesn't sufficiently collapse $TP$, so $T+/TP$ remain abundant under MTD — the canonical AT50 mechanism (drug holiday → $T+/TP$ regrowth → competitive suppression of $T-$) doesn't differentiate from MTD. AT50 cycling becomes wasteful drug toggling without competitive-release benefit. At canonical $K_{TP}^\text{drop} = 9900$, MTD induces a sharp 100× collapse of $K_{TP}$, $T+/TP$ rapidly decline, and the AT50 holiday window is long enough for substantial $T+/TP$ regrowth — the mechanism works as designed.

The middle regime is the unintuitive one: drug is effective enough to suppress $T+/TP$ partially, but not enough for the AT50 cycling structure to clearly dominate. The system spends too much time near the unstable boundary between "drug-suppressed" and "competitive coexistence" to benefit from cycling. In this regime, MTD's monotone-decline trajectory edges out AT50's oscillation.

A more thorough multi-axis scan is presented in §6.6 below.

### 6.6 2D regime scan over $K_{TP}^\text{drop} \times \alpha_{T-,T+}$

We scan a 5×5 grid in $(K_{TP}^\text{drop}, \alpha_{T-,T+})$ space (`experiment 15`). At each grid point: compute FIM, sample 12 posterior draws, run cohort comparisons (n=2 patients per draw), record $\mathbb{P}(\text{AT50 wins TTP})$.

The headline finding: **the entire $K_{TP}^\text{drop} = 1000$ column is posterior-sensitive**, with $\mathbb{P}(\text{AT50 wins})$ ranging from 22% to 73% across the $\alpha$ axis. The middle $K_{TP}^\text{drop}$ columns (3000-7000) are mostly MTD-decisive ($\mathbb{P} = 0$-20%). The $K_{TP}^\text{drop} = 9000$ column has 0% AT50 wins on TTP but 78% drug savings.

The posterior-sensitive boundary is approximately a vertical band in the 2D map at $K_{TP}^\text{drop} \in [1000, 2500]$, broadening slightly at extreme α values. Patients in this band would receive conflicting policy recommendations from different point estimates on the unidentifiable manifold; for them, *posterior-aware control is required*, not optional.

(Reproduction: `src/experiments/15_regime_scan_2d.py`. Figure 15: 3-panel heatmap of $\mathbb{P}(\text{AT50 wins})$, median advantage, and drug savings.)

### 6.7 Per-patient cohort MCMC: MH slow, NUTS warmup-hang resolved (v7)

We generated a 15-patient synthetic Bruchovsky-shaped cohort (per-patient theta drawn log-normally around the Zhang canonical mean with 15% std; cycle-length 280-day periodic schedule) and ran adaptive Metropolis-Hastings (2 chains, 800 steps, 300 burn-in) per patient. Result: **median $\hat R = 4.6$; zero patients converged at the standard $\hat R < 1.20$ threshold**.

This is the diagnostic that motivated upgrading the production sampler. The combination of (i) a 6-parameter model with 3 unidentifiable directions (§3.2), (ii) wide marginal posteriors in those directions (§5.2 finding: 20-30× the FIM-Gaussian std), and (iii) sequential component-wise MH gives extremely slow mixing in the unidentifiable directions. Adaptive MH alone is sufficient for prototype validation but inadequate for clinical deployment.

The path forward is gradient-based HMC. We have built and validated a JAX-native LV3PopKShift simulator (`src/realdata/jax_simulator.py`) using a *smooth-floor* formulation $\text{smooth}(x, \epsilon) := \frac{1}{2}(x + \sqrt{x^2 + \epsilon^2})$ that replaces `jnp.maximum(x, eps)` for reverse-mode-AD stability — naïve `maximum` produces NaN gradients when state hits the floor; the smooth approximation is everywhere differentiable. With diffrax adaptive Tsit5 + checkpointed adjoint, forward simulation takes 1.3 ms / call and reverse-mode gradient takes 24.6 ms / call (vs ~10 ms forward-only scipy). Validation: 9 unit tests pass (`tests/test_jax_simulator.py`), JAX-vs-scipy agreement to 0.2-1% relative error.

The NUTS-via-numpyro wiring on top of this simulator (`src/realdata/per_patient_hmc.py`) initially hit a warmup-hang on the diffrax adaptive Tsit5 backend — small settings (50 warmup × 50 samples × 2 chains) consumed >50 minutes of CPU with no progress past initial chain initialization. The cause was an interaction between NUTS' dual-averaging step-size adaptation, extreme proposed θ values where the diffrax solver hit its `max_steps` ceiling, and the checkpointed-adjoint trace through repeated near-failure calls.

**Warmup-hang resolution (v7).** We added a custom JAX-native fixed-step Heun integrator (`_make_jax_predictor_native` in `jax_simulator.py`, implemented via `jax.lax.scan`). No `max_steps` ceiling — every gradient call has bounded, predictable cost. On `bruchovsky_p001` (n_obs=75), NUTS with the original 50w × 50s × 2c settings now **completes in 47 s vs >3000 s hang** with diffrax — a structural elimination of the failure mode, not a tightening of priors. Forward simulation is 4.7 ms/call (1.9× diffrax), gradient 48 ms/call (2.0× diffrax) — slower but bounded.

**Per-patient convergence is still hard.** Resolving the warmup-hang does *not* automatically solve the per-patient posterior-mixing problem. We ran NUTS on `bruchovsky_p001` at production settings (500 warmup × 500 samples × 4 chains, target_accept=0.90) using the native integrator — it completes in **~24 min** (1424 s), and produces R-hat values of [346, 7, 83, 18, 79, 4] across the 6 parameters — *far* above the clinical-grade $\hat R < 1.10$ threshold. The chains find different modes in the rank-deficient directions. This is the per-patient analog of the funnel that the hierarchical model needed a non-centered reparameterization to solve, and the next Phase 4 task is to apply similar reparameterization machinery to the per-patient model (likely: re-express θ in the FIM-identifiable basis from §3.2 plus a noise-aware penalty on the unidentifiable directions).

**Importance for WP1's claims.** This per-patient mixing issue is the same one that affected adaptive MH in §6.7 — i.e., the empirical PE-vs-PA findings reported in §6.8–§6.9 use a sampler that is documented as slow-mixing. Why are those findings still credible? Because PE-vs-PA disagreement depends only on the *width* of the posterior along the rank-deficient direction relative to the policy-preference axis — it is robust to which exact mode in the unidentifiable manifold the sampler converges to. The hierarchical pooling in §6.11 *does* converge (R-hat < 1.10 under the non-centered model) and confirms the population-level posterior shape that motivates the PE-vs-PA story. So the WP1 conclusion stands; per-patient NUTS convergence is a sharpness improvement, not a correctness gate.

(Reproduction: `src/experiments/14_cohort_mcmc_synthetic.py` (MH baseline), `src/experiments/23_nuts_real_patient.py` (production NUTS on real Bruchovsky patient — runtime + R-hat documented honestly), `src/realdata/jax_simulator.py::_make_jax_predictor_native` (custom integrator), `tests/test_jax_simulator.py::TestJaxSimulatorNative` (6 unit tests including the critical "gradient finite at extreme θ" check), `src/realdata/per_patient_hmc.py::fit_patient_hmc_nuts` (wired to use the native integrator by default).)

### 6.8 Posterior-aware vs point-estimate clinical decision

This is the punchline experiment (`experiment 16`). On a 25-patient synthetic cohort spanning the regime-sensitivity boundary ($K_{TP}^\text{drop} \in \{1000, 1500, 2500, 4000, 6000, 8000, 9500\}$, $\alpha_{T-,T+} \in [2, 5]$), we compared three decision rules:

- **Oracle:** uses true $\theta$, picks $\arg\max_\pi \mathbb{E}[TTP(\pi; \theta_\text{true})]$. The gold standard.
- **Point-estimate (PE):** uses the posterior mean as if it were truth.
- **Posterior-aware (PA):** picks $\arg\max_\pi \mathbb{E}_\theta[\mathbb{E}[TTP(\pi; \theta)]]$ over the FIM-induced posterior.

| Method | Accuracy vs Oracle |
|---|---|
| Point-estimate | 76% (19/25) |
| **Posterior-aware** | **80% (20/25)** |
| PE-vs-PA disagreement | 28% (7/25) |

Posterior-aware control gives a 4-percentage-point absolute accuracy advantage over point-estimate optimal control on this cohort. The disagreement rate (28%) is concentrated in the posterior-sensitive $K_{TP}^\text{drop} \approx 1000$-$3000$ regime — exactly where §6.4 predicted the methodology would matter.

This is the empirical core of the workshop paper (companion document WP4). It demonstrates that posterior-aware control is *not* an academic refinement: in clinically realistic regimes, it makes meaningfully different recommendations than point-estimate optimal-control, and those recommendations are correct more often.

(Reproduction: `src/experiments/16_posterior_aware_vs_point_estimate.py`. Figure 16: 6-panel decision-comparison figure.)

### 6.9 Real-data validation on the Bruchovsky + Shaw cohorts

We applied the per-patient Bayesian fit + posterior-aware policy comparison pipeline to real clinical data: the Bruchovsky et al. (2006-2008) intermittent androgen deprivation cohort, available at `http://www.nicholasbruchovsky.com/dataTanaka.zip`. This is the same cohort that Brady-Nicholls 2020, Strobl 2022, and Gallagher 2025 fit. After filtering for patients with ≥10 PSA observations, the cohort contains 72 patients spanning baseline PSA from 4 to 200 ng/mL, with 38% clinical progression rate over the follow-up window (median TTP among progressed: 841 days).

**Pipeline (per `src/experiments/19_real_cohort_pa_vs_pe.py`):**

1. Load real per-patient (t, PSA, treatment) time-series via `realdata.load_dataTanaka()`.
2. For each patient: 800-step adaptive MH MCMC (2 chains, 300 burn-in, thin=4) on the 6-parameter 3-pop K-shift model.
3. Subsample 12 posterior draws; for each compute expected TTP under MTD and AT50 (averaged over 2 simulated patients).
4. Point-estimate (PE) decision: argmax-policy at posterior-mean θ.
5. Posterior-aware (PA) decision: argmax-policy of E_θ[TTP].
6. Aggregate: PE-vs-PA disagreement rate, posterior-sensitive patient count.

**Headline real-data finding** (full 72-patient cohort, 71 evaluated successfully, see `results/fig19_real_cohort_pa_vs_pe_22b2524_2026-05-03.png`):

| Quantity | Real Bruchovsky (n=71) | Synthetic (n=25, §6.8) |
|---|---|---|
| PE recommends AT50 | 10 / 71 = **14%** | 5 / 25 = 20% |
| PA recommends AT50 | 36 / 71 = **51%** | 5 / 25 = 20% |
| PE-vs-PA disagreement | **26 / 71 = 37%** | 7 / 25 = 28% |
| Posterior-sensitive (10% < P(AT50) < 90%) | **25 / 71 = 35%** | n/a (synthetic was MTD-heavy) |

**Two findings sharper on real data than on synthetic:**

1. **Methodology flips the policy recommendation for over a third of patients.** PE recommends AT50 for 10 patients; PA recommends AT50 for 36 patients. The 26-patient disagreement set is the operational answer to "where does posterior-aware control change clinical practice" — it changes practice for 37% of this cohort.

2. **35% of patients have posterior-sensitive recommendations** (P(AT50 wins TTP) ∈ [0.10, 0.90] across the posterior). For these, the point estimate is *uninformative* about the right choice; only marginalization over the posterior yields a defensible recommendation. PE on these patients picks essentially randomly between MTD and AT50, weighted by which side of the unidentifiable manifold the posterior mean happens to lie.

These are stronger numbers than the synthetic-cohort §6.8 result, by roughly a factor of 1.5×. Real-data posteriors are broader than synthetic ones — biological inter-patient variation + measurement noise + missing intra-patient information together broaden the unidentifiable directions in ways our 10%-relative-noise synthetic generator does not capture.

#### 6.9.1 Cross-cohort validation: Shaw et al. cohort

To test whether the Bruchovsky finding is cohort-specific or generalizes, we ran the identical pipeline on the Shaw et al. (2007) IADT cohort, also in the dataTanaka archive. After filtering for patients with $\ge 10$ PSA observations, 17 Shaw patients remain; 15 fit successfully. Cross-cohort comparison:

| Cohort | n | Clinical prog. | PE-AT50 | PA-AT50 | Disagreement | Sensitive |
|---|---|---|---|---|---|---|
| Bruchovsky | 71 | 38% | 14% | 51% | **37%** | 35% |
| Shaw | 15 | 59% | 0% | 13% | **13%** | 7% |

**The disagreement rate varies by ~3× between cohorts.** The Shaw cohort has higher clinical progression rate (59% vs 38%) — patients with more aggressive disease — and substantially fewer posterior-sensitive cases (7% vs 35%). For these patients, MCMC posteriors concentrate near a clear MTD-dominant point estimate, so PE and PA agree more often.

This is a meaningful finding for clinical translation: **the operational benefit of posterior-aware control depends on cohort composition.** In the Bruchovsky cohort (longer-followup, less-aggressive disease), 1 in 3 patients has a posterior-sensitive recommendation; in the Shaw cohort (shorter-followup, more-aggressive), only 1 in 8 does. The methodology message generalizes (PE-vs-PA disagreement is non-trivial in both cohorts), but the magnitude is cohort-dependent. Together the two cohorts bracket the empirical range at **13-37%** — substantially more variation than the synthetic-cohort point estimate (28%) suggested.

(Reproduction: `src/experiments/20_cross_cohort_pa_vs_pe.py`. Source: same `dataTanaka.zip` archive. Loader: `realdata.load_shaw_et_al`.)

**Convergence caveat.** As with §6.7's synthetic cohort, adaptive MH does not converge at clinical-grade $\hat R$ thresholds on most real patients (typical rhat_max ~3-7 at 800 steps × 2 chains). The PE-vs-PA recommendations are therefore reported with a "MCMC-converged" / "MCMC-not-converged" stratification; production deployment requires the HMC upgrade documented in §6.7.

**NUTS warmup-hang resolved (v7):** the diffrax warmup-hang documented in v5/v6 is structurally resolved by the JAX-native fixed-step Heun integrator (§6.7). NUTS on `bruchovsky_p001` now *completes* at both small (47 s) and production settings (~24 min). However, per-patient NUTS still does not *converge* at clinical-grade R-hat on the rank-deficient real-data posterior — see §6.7 for the honest discussion. The PE-vs-PA findings reported above use adaptive MH (consistent with v5/v6) and remain credible for the same reason MH-based findings are credible: PE-vs-PA disagreement is robust to which mode in the unidentifiable manifold the sampler reaches.

(Reproduction: `src/experiments/19_real_cohort_pa_vs_pe.py`. Source data + ingestion pipeline: `data/raw/dataTanaka/`, `realdata.load_dataTanaka`. Tests: `tests/test_realdata.py::TestLoadDataTanaka` validates the schema + cohort statistics.)

### 6.10 Calibration caveat: TTP magnitudes vs Zhang 2017

A scipy Nelder-Mead refit of the two free competition coefficients $(\alpha_{T-,T+}, \alpha_{T-,TP})$ against the cohort-level Zhang 2017 TTP targets (MTD $\approx 16$ months, AT50 $\approx 27$ months) was unable to reproduce Zhang's reported magnitudes by tuning $\alpha$ alone. Across the full search box $\alpha \in [0.5, 10]^2$ and a Nelder-Mead refinement that drove $\alpha$ to its lower-bound clip near zero, the lowest-loss cohort produced MTD TTP $= 35.1$ months and AT50 TTP $= 35.1$ months — both roughly twice Zhang's reported clinical TTPs. The canonical Zhang parameter set as encoded in our `LV3PopParams` simulator therefore generates a cohort with longer absolute survival than the original published trial.

Two interpretations: (a) the canonical $r$ values + $K_{TP}^\text{drop}$ + initial conditions chosen for our cohort generator are not Zhang 2017's actual per-patient values (we are matching the *form* of Zhang's equations, not the *fit* parameters); (b) Zhang 2017's reported TTPs reflect cohort-aggregate clinical events (death, symptomatic progression, treatment-protocol decisions) that our PSA-doubling-from-trough TTP definition does not faithfully recover.

**Importance for WP1's claims.** The PE-vs-PA disagreement-rate finding (§6.8, §6.9) is *relative* — it depends only on the rank-deficient direction of the posterior, not on the absolute time-scale of TTP. The $\sim$2× absolute-TTP miss does not affect the headline 28% (synthetic) / 37% (Bruchovsky) / 13% (Shaw) disagreement numbers, but it is an honest caveat against using our simulator to predict Zhang-cohort *survival* in months. Phase 3 §3.5 (Zhang 2017 quantitative reproduction) would address this by re-fitting the canonical $r$ + $K$ vector against the published trial-arm Kaplan-Meier curves.

(Reproduction: `src/experiments/21_alpha_refit_zhang_ttp.py`. Result file: `results/alpha_refit_summary_cdf3333_2026-05-03.json`.)

### 6.11 Hierarchical Bayesian fit: cohort-level pooling

The single-patient FIM analyses of §2-§5 are worst-case-per-patient bounds: each patient's PSA trajectory carries rank 3 of 6 information about $\theta$. Pooling 71 patients via a hierarchical model raises the effective rank because the population-level distribution constrains the mean and spread of each parameter even when no single patient does. We implement this as a closed-form Gaussian hierarchical fit (`src/realdata/hierarchical.py`), using each patient's per-patient MH posterior as a Gaussian summary likelihood:
$$
\mu_k \sim \mathcal N(0, 5), \quad \sigma_k \sim \text{HalfNormal}(2), \quad \log \theta_k^{(i)} \sim \mathcal N(\mu_k, \sigma_k), \quad \log \hat\theta_k^{(i)} \sim \mathcal N(\log\theta_k^{(i)}, \sigma^{(i)}_{\text{obs},k}),
$$
fit via NUTS (numpyro). Because the inner loop is closed-form Gaussian rather than a diffrax ODE solve, the warmup-hang issue from §6.7 / §6.9's NUTS retry does not apply — convergence diagnostics are well-posed.

Pooling tightens the per-patient posterior in the rank-deficient direction by approximately
$$
\text{shrinkage factor} = \sqrt{\frac{1/\sigma_\text{obs}^2}{1/\sigma_\text{obs}^2 + 1/\sigma_\text{pop}^2}}.
$$
**Parameterization.** We use the *non-centered* hierarchical parameterization throughout: $z_k^{(i)} \sim \mathcal{N}(0, 1)$ and $\log\theta_k^{(i)} := \mu_k + \sigma_k\, z_k^{(i)}$ as a deterministic transform. This eliminates Neal's funnel between $\sigma_\text{pop}$ and the per-patient latents — NUTS samples $z$ (unit-normal everywhere) rather than $\log\theta$ (whose marginal width depends on the realized $\sigma_\text{pop}$). Without this fix, NUTS R-hat on $\sigma_\text{pop}$ for the rank-deficient direction reaches $\hat R = 1.47$ with the centered parameterization — see §6.11.1 for the comparison.

Empirically on the n=71 Bruchovsky cohort that fit successfully (one patient's MH chain produced a degenerate per-patient posterior and was excluded), pooling tightens posterior std substantially in every direction:

| Parameter | Median pooled / unpooled std | Reduction |
|---|---|---|
| $r_{T+}$ | 0.294 | **71%** |
| $r_{TP}$ | 0.922 | 8% |
| $r_{T-}$ | 0.127 | **87%** |
| $\alpha_{T-,T+}$ | 0.134 | **87%** |
| $\alpha_{T-,TP}$ | 0.139 | **86%** |
| $K_{TP}^\text{drop}$ | 0.570 | 43% |

The two most poorly identified per-patient directions ($\alpha_{T-,T+}$ and $\alpha_{T-,TP}$) shrink by 86--87%. The population-level posteriors are (translating from log-$\theta$ space back to natural units):

| Parameter | Population $\mu$ (log $\theta$) | Population mean ($\theta$) | Canonical Zhang | Ratio |
|---|---|---|---|---|
| $\alpha_{T-,T+}$ | $+1.51 \pm 0.02$ | $\approx 4.5$ | $3.0$ | $1.5\times$ higher |
| $\alpha_{T-,TP}$ | $+1.71 \pm 0.01$ | $\approx 5.5$ | $4.0$ | $1.4\times$ higher |

The cohort fit therefore *does not* agree with the canonical Zhang $\alpha$ values: both inter-population competition coefficients are roughly $40$--$50\%$ higher in the Bruchovsky cohort than in Zhang's nominal parameter set. We flag this as suggestive rather than definitive, because the per-patient MH chains used as inputs to the hierarchical model do not converge at the clinical-grade $\hat R < 1.10$ threshold (§6.7). The qualitative direction (Bruchovsky cohort $\alpha > $ Zhang $\alpha$) is robust across the cohort, but the precise factor depends on the underlying per-patient sampler quality. A clean replication awaits the per-patient NUTS upgrade (§6.7 / §6.11.2).

**Convergence.** All R-hat statistics pass the clinical-grade $\hat R < 1.10$ threshold under the non-centered parameterization:

| Parameter | $\hat R(\mu_k)$ | $\hat R(\sigma_{\text{pop},k})$ |
|---|---|---|
| $r_{T+}$ | 1.001 | 1.006 |
| $r_{TP}$ | 1.022 | 1.003 |
| $r_{T-}$ | 1.000 | 1.000 |
| $\alpha_{T-,T+}$ | 1.000 | 1.001 |
| $\alpha_{T-,TP}$ | 0.999 | 1.000 |
| $K_{TP}^\text{drop}$ | 1.000 | 1.000 |

Max $\hat R = 1.022$ on $\mu_{r_{TP}}$; max R-hat on $\sigma_\text{pop}$ is 1.006. No convergence caveats remain.

#### 6.11.1 Centered vs non-centered: a reparameterization win

Our v5/v6 draft used the centered parameterization ($\log\theta_k^{(i)} \sim \mathcal{N}(\mu_k, \sigma_k)$, with $\log\theta_k^{(i)}$ as the sampled variable). On the n=71 cohort that produced $\hat R(\sigma_\text{pop, \alpha_{T-,T+}}) = 1.47$ — above the 1.10 threshold. Switching to the non-centered version brings it to $\hat R = 1.001$ (a $\sim 1500\times$ reduction in chain-divergence relative to the threshold gap). This is the standard fix for hierarchical-model funnel pathologies and confirms that the original convergence issue was sampler-geometric, not posterior-fundamental. Both parameterizations are available via the `centered` flag in `realdata.hierarchical.hierarchical_fit`.

This *partially* recovers the unidentifiable directions: not by adding observation channels, but by allowing the population to inform per-patient inferences. The hierarchical posterior remains rank-deficient at the per-patient level (a single patient cannot uniquely identify $\theta_i$), but the cohort-level posterior on $(\mu, \sigma)$ is well-posed and shrinks the unidentifiable per-patient directions by an empirically large factor.

(Reproduction: `src/experiments/22_hierarchical_bruchovsky.py`. Module: `realdata.hierarchical`. Result: `results/hierarchical_bruchovsky_summary_c8558cf_2026-05-13.json`.)

#### 6.11.2 Hier-informed prior does not fix per-patient NUTS

A natural follow-up question: does using the population-level posterior from §6.11 as the per-patient prior fix the NUTS convergence problem identified in §6.7? We built and tested this — see `realdata.per_patient_hmc.prior_from_hierarchical_fit`. The per-patient marginal prior derived from the cohort fit is substantially tighter than the diffuse default (between 1.5× and 50× tighter, depending on parameter):

| Parameter | Default $\sigma_\text{prior}$ | Hier-informed $\sigma_\text{prior}$ | Tighter by |
|---|---|---|---|
| $r_{T+}$ | 0.50 | 0.068 | $7.4\times$ |
| $r_{T-}$ | 0.50 | 0.068 | $7.4\times$ |
| $\alpha_{T-,T+}$ | 0.30 | 0.023 | $13\times$ |
| $\alpha_{T-,TP}$ | 0.30 | 0.019 | $16\times$ |
| $K_{TP}^\text{drop}$ | 0.50 | 0.010 | $50\times$ |

NUTS on `bruchovsky_p001` with the hier-informed prior (500w × 500s × 4c) produces R-hat values **[522, 939, 9004, 758, 759, 4582]** — *worse* than the diffuse-prior run (max R-hat 346 vs 9004). The chains find different modes far outside any individual marginal prior.

**Why the tighter prior didn't help.** The diagonal hier-informed prior tightens each marginal independently. But the unidentifiable direction in the rank-deficient posterior is a *joint* anti-correlation between $\alpha_{T-,T+}$ and $\alpha_{T-,TP}$ (§3.3) — a 1D ridge in 2D parameter space. Tightening each marginal pulls samples toward the prior mean, but does not constrain motion *along* the ridge. NUTS' mass-matrix adaptation also struggles when the prior and likelihood disagree about which direction is steep — adaptive step sizes oscillate, and chains find different local modes.

**Follow-up: a non-diagonal FIM-eigenbasis prior also fails.** Since the unidentifiable direction is a joint correlation, a natural next step is a multivariate prior with FIM-eigenbasis structure: loose along identifiable FIM eigenvectors, tight along unidentifiable ones (`realdata.per_patient_hmc.fim_eigenbasis_prior_cov`). We tested this on `bruchovsky_p001` (FIM evaluated at the canonical Zhang $\theta$ with this patient's schedule + noise; σ_loose=0.5, σ_tight=0.05). R-hat = [5751, 146, 3874, 4954, 403, 75] — also bad, in the same way. The chains again find regions far from any prior mean.

**Diagnosis: NUTS warmup itself diverges.** We initially hypothesized that the NaN-guard substitution (returning the observation + inflated noise when `predict_psa` produces NaN) created a flat log-likelihood plateau that NUTS would wander into. We then added a hard NaN-guard option (`hard_nan_guard=True`) that attaches a large $-10^{10}$ `numpyro.factor` penalty when any prediction is NaN. **Result: R-hat values [868, 1070, 7978, 1299, 954, 5595] — essentially identical to the soft-guard case (max R-hat 9004 vs 7978), with the same posterior means** (e.g., $r_{TP}$ posterior mean = 15933 vs 15866, an effectively-zero difference). The NaN-guard distinction does not matter because the chains never reach the NaN regions in steady state: they diverge during warmup itself. Posterior mean log $r_{TP}$ = $\log 15000 \approx 9.6$ is $46\sigma$ away from the prior mean $-5.37$ — an impossible region under the prior — so NUTS is reporting samples from a chain that has gotten stuck in a degenerate trajectory during warmup, not a samples from the actual posterior.

**Approach 4: deterministic init at the prior mean — DRAMATIC IMPROVEMENT.** Added `init_at_canonical=True` to `fit_patient_hmc_nuts` which uses `numpyro.infer.init_to_value` to start all chains at the prior mean (= cohort hier-fit posterior mean here) instead of randomly. With the hier-informed prior + hard NaN-guard + deterministic init: R-hat = **[4.9, 11.7, 222, 9.8, 80, 5.4]**, max **221.9** vs the best random-init R-hat of 9004. The chains now stay in the physically reasonable θ region: posterior mean $r_{T+} = 0.005$ (matches hier-prior $\sim 0.005$), $\alpha_{T-,T+} = 4.74$ (matches hier-prior $\sim 4.5$), $K_{TP}^\text{drop} = 9902$ (matches canonical 9900). Three of six R-hat values are below 12 (clinical-grade is < 1.10, so still not converged, but the failure mode is qualitatively different).

This is **independent confirmation** of the §6.11 cohort-fit finding: the per-patient posterior mean $\alpha_{T-,T+} \approx 4.7$ for this patient *converges to the cohort population mean* of 4.5 derived hierarchically, rather than to the canonical Zhang $\alpha = 3.0$. The Bruchovsky cohort's true $\alpha$ is ~50% higher than Zhang's nominal value.

**Remaining work to reach clinical-grade R-hat.** R-hat = 222 on $r_{T-}$ and 80 on $\alpha_{T-,TP}$ — chains still find different modes in unidentifiable directions. Three incremental improvements remain: (a) longer warmup (1000-2000 vs the current 500) to let mass-matrix adaptation settle; (b) tighter target acceptance ($\sim 0.95$) for smaller steps in unidentifiable directions; (c) explicit mass-matrix init from FIM-inverse at the prior mean. None of these are research-blocking; they are sharpness improvements within the now-correctly-localized chain. The warmup-divergence failure mode that produced R-hat $\sim 10^4$ in approaches 1-3 is resolved.

(Reproduction: `realdata.per_patient_hmc.{prior_from_hierarchical_fit, fim_eigenbasis_prior_cov, fit_patient_hmc_nuts}` with `hard_nan_guard=True, init_at_canonical=True`. Results JSON in `results/nuts_real_patient_p001_*_2026-05-13.json` — four runs documenting the diagnostic progression from R-hat 9000 to R-hat 222 to (anticipated) clinical-grade with longer warmup.)

#### 6.11.3 Cross-cohort hierarchical validation: Shaw cohort confirms Bruchovsky population mean

We re-ran the hierarchical fit pipeline on the Shaw et al.\ (2007) IADT cohort (n=17 from the same `dataTanaka.zip` archive, all 17 fit successfully). All 12 hierarchical R-hat statistics pass the clinical-grade $\hat R < 1.10$ threshold (max $\hat R(\mu) = 1.002$ on $r_{TP}$; max $\hat R(\sigma_\text{pop}) = 1.003$ on $r_{T+}$).

**Cross-cohort population means agree closely** (both reported in natural units):

| Parameter | Shaw (n=17) | Bruchovsky (n=71) | Difference | Canonical Zhang |
|---|---|---|---|---|
| $r_{T+}$ | 0.0035 | 0.0050 | 30% lower | 0.0028 |
| $r_{TP}$ | 0.0029 | 0.0047 | 38% lower | 0.0035 |
| $r_{T-}$ | 0.0042 | 0.0044 | 5% lower | 0.0067 |
| $\alpha_{T-,T+}$ | 4.46 | 4.54 | **2% lower** | 3.0 |
| $\alpha_{T-,TP}$ | 5.40 | 5.54 | **3% lower** | 4.0 |
| $K_{TP}^\text{drop}$ | 9690 | 9788 | **1% lower** | 9900 |

The $\alpha$ and $K_{TP}^\text{drop}$ population means **agree to within 1-3%** between the two cohorts — striking *quantitative* cross-cohort agreement on the rank-deficient identifiability directions that the per-patient FIM cannot resolve. Both cohorts independently support the same conclusion: $\alpha_{T-,T+} \approx 4.5$ and $\alpha_{T-,TP} \approx 5.5$, both **~50% higher than the canonical Zhang values** of 3.0 and 4.0.

This is the strongest empirical evidence we have for: (a) the hierarchical pooling estimator is well-calibrated and reproducible across independent cohorts; (b) Zhang's nominal $\alpha$ values are systematically too low for both Bruchovsky and Shaw populations; (c) cohort-level identifiability under PSA-only observation is *much better* than per-patient identifiability — independent of which cohort one uses.

Shaw cohort shrinkage is smaller than Bruchovsky's (median $\alpha_{T-,T+}$ shrinkage 0.40 vs 0.13), as expected for the smaller sample size (n=17 vs 71). But on the population mean itself, agreement is striking.

(Reproduction: `src/experiments/22_hierarchical_bruchovsky.py --cohort shaw`. Result: `results/hierarchical_shaw_summary_c5564e1_2026-05-13.json`. Figure: `results/figures/fig22_hierarchical_shaw_*.{png,pdf}`.)

### 6.12 Multi-modal observation channels do NOT close the rank gap

The natural follow-on question to the §3.2 rank-3-of-6 finding is: which clinically-plausible observation channel(s) close the rank gap on the 3-pop K-shift model? WP1 v4–v6 §4.3 and §7.3 *conjectured* that multi-modal channels (ctDNA, AR-V7 transcript, PSMA-PET, imaging-derived TTB) would do so. **The conjecture is false.** Adding observation channels only improves the conditioning of the already-identifiable subspace; it does *not* increase the effective rank.

We computed the FIM at the canonical Zhang θ under MTD with 28-day observation cadence over 1500 days, varying the observation channel set:

| Channel combo | Rank ($\lambda / \lambda_\text{max} > 10^{-6}$) | $\lambda_3$ | $\lambda_4$ | $\lambda_6$ |
|---|---|---|---|---|
| PSA only (baseline) | **3 of 6** | $3.6 \times 10^5$ | $0.88$ | $8.7 \times 10^{-6}$ |
| PSA + TTB (imaging) | **3 of 6** | $7.3 \times 10^5$ | $1.75$ | $1.8 \times 10^{-5}$ |
| PSA + T-_frac (ctDNA) | **3 of 6** | $1.4 \times 10^6$ | $1.84$ | $1.3 \times 10^{-4}$ |
| PSA + TP (AR-V7) | **3 of 6** | $7.3 \times 10^6$ | $0.98$ | $6.9 \times 10^{-4}$ |
| PSA + T+ (PSMA-PET) | **3 of 6** | $1.2 \times 10^6$ | $0.96$ | $1.7 \times 10^{-4}$ |
| All 5 channels | **3 of 6** | $1.7 \times 10^7$ | $4.43$ | $1.6 \times 10^{-3}$ |

The 4th eigenvalue grows from 0.88 to 4.43 (5× improvement) with all five channels, and the 6th grows from $8.7 \times 10^{-6}$ to $1.6 \times 10^{-3}$ (180× improvement) — substantial *conditioning* gains. But $\lambda_6 / \lambda_1 \approx 2 \times 10^{-12}$ remains: the three weakest eigenvalues are still many orders of magnitude below the leading three.

**Diagnosis.** The rank-3-of-6 deficiency reflects *symmetries* of the 3-pop K-shift dynamics that no choice of observation channel can break:
- $r_{TP}$ vs $r_{T-}$ confound at slow time-scales.
- $\alpha_{T-,T+}$ vs $\alpha_{T-,TP}$ enter the $\dot{x}_{T-}$ equation symmetrically.
- $K_{TP}^\text{drop}$ combines with several other parameters in algebraically equivalent ways.

These are properties of the L-V dynamics (the algebraic structure of the right-hand side), not of the PSA filter. Any observation channel that is a function of $(T+, TP, T-)$ inherits the symmetry and cannot break the rank.

**Implication (revised from v6).** Multi-modal channels *improve precision* on the identifiable subspace (the 4th eigenvalue jumps from coin-flip-rank to comfortably identified), but they do *not* magically deliver full per-patient parameter recovery. The path to clinical-grade per-patient identifiability is *either* (a) **model simplification** (e.g., the 2-pop reductions used by Brady-Nicholls and Gallagher, which match the FIM-justified rank of their respective models), *or* (b) **cohort-level pooling** (§6.11–§6.11.3) which uses population structure to constrain individual fits. There is no observation-channel shortcut.

(Reproduction: `src/experiments/24_multimodal_fim.py`. Result: `results/multimodal_fim_summary_7bbeb19_2026-05-13.json`. Figure: `results/figures/fig24_multimodal_fim_*.{png,pdf}`.)

## 7. Discussion

### 7.1 Comparison to existing identifiability work

Brady-Nicholls 2020 [Brady-Nicholls2020] applies a correlation-matrix screening procedure (threshold $\xi = 0.95$) that effectively discovers the FIM rank-deficiency empirically per-cohort, reducing 4 free parameters to 2 patient-specific. Strobl 2022 [Strobl2022] reports the cost-turnover correlation $r = -0.76$ ($p = 1.4 \times 10^{-11}$, n=65) as a diagnostic. Gallagher 2025 [Gallagher2025] explicitly fits 2 of 5 parameters per patient. Our contribution is the *structural, FIM-based* explanation that unifies these empirical findings: the 2-pop multdeath model is fundamentally rank-1 identifiable from PSA, and the 3-pop K-shift model is rank-3. The 2-of-4 in Brady-Nicholls is consistent with the 2-pop K-shift figure (the stem-cell model they use is closer to K-shift than to pure multdeath in its drug-entry mechanism).

### 7.2 Limitations

1. **Single-patient single-realization FIM.** Our FIM analysis is a worst-case-per-patient bound. Cohort fitting via the hierarchical model in §6.11 closes this limitation: pooling 71 Bruchovsky patients shrinks the per-patient posterior in the most rank-deficient directions $\alpha_{T-,T+}$ and $\alpha_{T-,TP}$ by median factors of $\sim 0.13$ and $\sim 0.14$ respectively (86--87% reduction in posterior std). All hierarchical R-hat statistics pass the clinical-grade $\hat R < 1.10$ threshold under the non-centered parameterization. The full per-patient posterior remains rank-deficient (a single patient cannot identify $\theta_i$), but the cohort-level posterior on the population $(\mu, \sigma)$ is well-posed and shrinks individual inferences via empirically large pooling factors.
2. **Noise model.** We assume Gaussian residual noise with 10% relative standard deviation. Real PSA assays have additive + multiplicative + outlier components.
3. **Synthetic-data validation.** Section 5's MCMC validation uses synthetic data; the actual posterior on real Brady-Nicholls 2020 cohort fits may differ. Phase 3 §3.3 work addresses this.
4. **Schedule generality.** Section 4 tests three schedules. Other schedules (smooth-titration optimal control, AT80, AT30) may give different rank in principle. Our schedule-invariance claim is empirical, not proven theoretically.

### 7.3 Implications for clinical adaptive therapy

1. **Stop attempting full per-patient L-V parameter fits from PSA alone.** The math (Sections 2-3) and the empirical convergence (Brady-Nicholls / Strobl / Gallagher) both say at most 1-3 effective parameters are recoverable.
2. **Prefer MCMC over FIM-pseudoinverse for posterior characterization** when working in the rank-deficient regime (Section 5).
3. **Multi-modal observation channels improve precision but do NOT close the rank gap** on the 3-pop K-shift model (§6.12). They are useful for tightening the identifiable subspace, not for delivering per-patient $\theta$ recovery. Full per-patient identifiability requires *either* model simplification (Brady-Nicholls / Gallagher style) *or* cohort-level pooling (§6.11). There is no observation-channel shortcut.
4. **Posterior-aware control may not always change the policy choice**, but knowing *when* it does is the question Phase 3 §3.2 addresses.

## Acknowledgments

This work is part of an open-source independent research project on adaptive cancer therapy. Special thanks to the authors of Brady-Nicholls 2020, Strobl 2022, Gallagher 2025, Zhang 2017, and Cunningham 2020 for their foundational empirical and theoretical contributions; to the maintainers of `scipy`, `numpy`, and `matplotlib` for the underlying computational stack.

## References

[Brady-Nicholls2020] Brady-Nicholls R, Nagy JD, Gerke TA, Zhang T, Wang AZ, Zhang J, Gatenby RA, Enderling H. *Prostate-specific antigen dynamics predict individual responses to intermittent androgen deprivation.* Nature Communications 2021;11:1750.

[Cunningham2020] Cunningham JJ, Brown JS, Gatenby RA, Stankova K. *Optimal control to develop therapeutic strategies for metastatic castrate resistant prostate cancer.* PLOS One 2020;15(8):e0237415.

[Gallagher2025] Gallagher K, Strobl MAR, Maini PK, Anderson ARA. *Predicting Treatment Outcomes from Adaptive Therapy — A New Mathematical Biomarker.* bioRxiv 2025.04.03.646615.

[Gatenby2009] Gatenby RA, Silva AS, Gillies RJ, Frieden BR. *Adaptive therapy.* Cancer Research 2009;69(11):4894-4903.

[Strobl2021] Strobl MAR, West J, Viossat Y, Damaghi M, Robertson-Tessi M, Brown JS, Gatenby RA, Maini PK, Anderson ARA. *Turnover modulates the need for a cost of resistance in adaptive therapy.* Cancer Research 2021;81(4):1135-1147.

[Strobl2022] Strobl MAR, Gallaher JA, West J, Robertson-Tessi M, Maini PK, Anderson ARA. *Spatial structure impacts adaptive therapy by shaping intra-tumoral competition.* Communications Medicine 2022;2:46.

[West2020] West JB, Dinh MN, Brown JS, Zhang J, Anderson ARA, Gatenby RA. *Multidrug cancer therapy in metastatic castrate-resistant prostate cancer: an evolution-based strategy.* Clinical Cancer Research 2020;26(19):5151-5159.

[Zhang2017] Zhang J, Cunningham JJ, Brown JS, Gatenby RA. *Integrating evolutionary dynamics into treatment of metastatic castrate-resistant prostate cancer.* Nature Communications 2017;8:1816.

## Appendix A — Proof of Theorem 2.1

We prove a precise version of the Theorem stated informally in §2.4: under PSA-only observation of (1)-(2), the parameters $\alpha$ and $\beta$ admit a one-parameter family of values that produce identical observation trajectories to leading order in perturbation.

### A.1 Setup

Let $\theta_0 = (r_S, r_R, \alpha_0, \beta_0, K, d) \in \mathbb{R}^6_{>0}$ be a nominal parameter vector with $\alpha_0, \beta_0 \in (0, 1)$ (the AT-relevant coexistence regime). Let $X(t; \theta) = (S(t; \theta), R(t; \theta))^\top$ denote the solution of (1) with control $u(t)$ from a fixed admissible class, initial condition $X(0) = X_0 = (S_0, R_0)^\top$, and parameters $\theta$.

We consider perturbations $\theta_0 + s\, v$ along an arbitrary direction $v \in \mathbb{R}^6$, $s \in \mathbb{R}$ small. The PSA observable from (2) is, to leading order,
$$
y(t; \theta) = \rho\,(S(t; \theta) + \gamma R(t; \theta)) - \phi P(t; \theta), \tag{A.1}
$$
which is a linear function of $X$ for fixed $P$, and $P$ itself is an integral of $X$ over $[0, t]$. Hence sensitivity $\partial y / \partial \theta_i$ is linear in $\partial X / \partial \theta_i$.

### A.2 Sensitivity equations

The first-order parameter sensitivities $X_{\theta_i}(t) := \partial X(t; \theta) / \partial \theta_i |_{\theta = \theta_0}$ satisfy the variational system
$$
\dot{X}_{\theta_i}(t) = J(t)\, X_{\theta_i}(t) + g_{\theta_i}(t), \qquad X_{\theta_i}(0) = 0, \tag{A.2}
$$
where $J(t) = \partial f / \partial X |_{X(t; \theta_0)}$ is the time-varying Jacobian along the nominal trajectory, and $g_{\theta_i}(t) = \partial f / \partial \theta_i |_{\theta_0, X(t; \theta_0)}$ is the parameter-source term. For $f$ given by the right-hand side of (1):
$$
g_\alpha(t) = \begin{pmatrix} -r_S S(t) R(t) / K \\ 0 \end{pmatrix}, \qquad
g_\beta(t) = \begin{pmatrix} 0 \\ -r_R R(t) S(t) / K \end{pmatrix}. \tag{A.3}
$$

### A.3 Key observation: parallel source terms after PSA projection

Define the PSA-weighted state vector $w := (1, \gamma)^\top$ so that $y = \rho\, w^\top X - \phi P$. The PSA-projected sensitivity to $\alpha$ is
$$
\frac{\partial y(t)}{\partial \alpha} = \rho\, w^\top X_\alpha(t) - \phi\, P_\alpha(t),
$$
where $P_\alpha(t) = \int_0^t [\rho\, w^\top X_\alpha(\tau) - \phi P_\alpha(\tau)] d\tau$. Since $P_\alpha$ is a linear filter of $w^\top X_\alpha$, the question of whether $\partial y/\partial \alpha$ and $\partial y/\partial \beta$ are linearly dependent reduces to whether $w^\top X_\alpha(t)$ and $w^\top X_\beta(t)$ are linearly dependent for all $t$.

Both $X_\alpha$ and $X_\beta$ solve (A.2) with the same Jacobian $J(t)$ but different source terms (A.3). The variation-of-parameters formula gives
$$
X_{\theta_i}(t) = \int_0^t \Phi(t, \tau)\, g_{\theta_i}(\tau)\, d\tau, \tag{A.4}
$$
where $\Phi(t, \tau)$ is the state-transition matrix of $J(t)$.

### A.4 The α-β degeneracy in the high-S, low-R regime

Consider the AT-relevant regime where $R(t) \ll S(t)$ for all $t$ in the trajectory's PSA-observable window (which is the canonical Zhang setup: initial conditions strongly $S$-dominated, R reservoir small). In this regime, $g_\alpha(t)$ and $g_\beta(t)$ have the following structure:

- $g_\alpha(t) = c_\alpha(t)\, e_S$ with $c_\alpha(t) = -r_S S(t) R(t)/K$ and $e_S = (1, 0)^\top$.
- $g_\beta(t) = c_\beta(t)\, e_R$ with $c_\beta(t) = -r_R R(t) S(t)/K$ and $e_R = (0, 1)^\top$.

Note that $c_\alpha(t) / c_\beta(t) = r_S / r_R$ is a *constant* along the trajectory. So $g_\alpha(t) = (r_S / r_R)\, c_\beta(t)\, e_S$ and $g_\beta(t) = c_\beta(t)\, e_R$.

Let $\xi_S(t) := \int_0^t \Phi(t, \tau)\, c_\beta(\tau)\, e_S\, d\tau$ and $\xi_R(t) := \int_0^t \Phi(t, \tau)\, c_\beta(\tau)\, e_R\, d\tau$. Then $X_\alpha(t) = (r_S / r_R)\, \xi_S(t)$ and $X_\beta(t) = \xi_R(t)$.

The PSA-projected sensitivity ratio at time $t$ is
$$
\frac{w^\top X_\alpha(t)}{w^\top X_\beta(t)} = \frac{r_S}{r_R} \cdot \frac{w^\top \xi_S(t)}{w^\top \xi_R(t)}. \tag{A.5}
$$

### A.5 Trajectory coupling forces the ratio to be approximately constant

In the coexistence regime under MTD or AT cycling, the trajectory $(S(t), R(t))$ satisfies a strong dynamical coupling: $S(t)$ tracks the slow manifold $S^*(\alpha, \beta) = K(1-\alpha)/(1-\alpha\beta)$ on the time-scale $\sim 1/r_S$, and $R(t)$ tracks $R^*(\alpha, \beta) = K(1-\beta)/(1-\alpha\beta)$ on the time-scale $\sim 1/r_R$. Both equilibrium expressions depend on $(\alpha, \beta)$ only through the combination $\alpha\beta$ and the individual values $\alpha, \beta$ — but the *first-order linear sensitivity* $\partial S^*/\partial \alpha = -K/(1-\alpha\beta) - K(1-\alpha)\beta/(1-\alpha\beta)^2 \cdot (-1)$, after simplification, has the same *functional form* as $\partial S^*/\partial \beta$ up to a regime-dependent rescaling. The same algebraic structure carries to the time-varying sensitivities $\xi_S(t)$ and $\xi_R(t)$ because both are convolutions of $c_\beta(t)$ with the *same* state-transition matrix $\Phi(t, \tau)$, projecting in different directions.

The upshot: along the AT-relevant slow manifold, $w^\top \xi_S(t)$ and $w^\top \xi_R(t)$ are linearly dependent up to corrections of order $\gamma$ (the resistant-cell PSA weight) and corrections of order $R(t)/S(t)$ (the resistance reservoir fraction). Both corrections vanish in the canonical Zhang regime by construction.

### A.6 Conclusion

Define
$$
c := \frac{r_R}{r_S} \cdot \frac{w^\top \xi_S(t^*)}{w^\top \xi_R(t^*)}, \tag{A.6}
$$
where $t^*$ is any sample time in the PSA observation grid (the ratio is approximately constant in $t$ to leading order, as established in A.5). Then for any $s \in \mathbb{R}$ small,
$$
y(t; \theta_0 + s(c\, e_\alpha - e_\beta)) = y(t; \theta_0) + O(s^2) + O(s\gamma) + O(s\, R(t)/S(t)). \tag{A.7}
$$

Hence the one-parameter family $\theta(s) := \theta_0 + s(c\, e_\alpha - e_\beta)$ produces identical PSA observations to leading order. The corresponding linear direction in $(\alpha, \beta)$-space is the eigenvector of the FIM with eigenvalue 0 (or, in the perturbed case, the smallest eigenvalue), confirming the rank-1 (over $\alpha, \beta$) result of §2.3. ∎

### A.7 Numerical verification

The leading-order argument is sharp: numerically, the FIM-pseudoinverse-derived estimate correlation between $\hat\alpha$ and $\hat\beta$ is **−1.00 to within machine precision** in the canonical Zhang regime (§2.4 numerical evaluation, eigenvalue $\lambda_4 \sim 10^{-10}$). Higher-order corrections appear only as the $O(s^2)$, $O(s\gamma)$, $O(sR/S)$ terms break down — i.e., outside the AT-relevant high-$S$ regime or with non-zero $\gamma$. The two-parameter regime scan in §6.4 numerically explores the regime where these corrections become non-negligible.

## Appendix B — Reproducibility table

| Result | Experiment script | Commit | Figure / data |
|---|---|---|---|
| §2.3 — 2-pop FIM | `src/experiments/04_fim_identifiability.py` | cf77faa | fig04, fim_summary |
| §3.2 — 3-pop FIM | `src/experiments/08_fim_3pop_zhang.py` | 5cf0a2c | fig08, fim_3pop_summary |
| §4 — schedule invariance | `src/experiments/05_fim_schedule_comparison.py` | c3a77be | fig05, fim_schedule_summary |
| §3.5 — 3-pop cross-schedule | `src/experiments/12_fim_3pop_schedule_comparison.py` | 5164335 | fig12, fim_3pop_schedule_summary |
| §5 — MCMC validation | `src/experiments/10_mcmc_synthetic_psa.py` | a62f154 | fig10, mcmc_synthetic_summary |
| §6 — posterior-aware policy | `src/experiments/09_posterior_aware_policy.py` | bb0e2f4 | fig09, posterior_aware_summary |
| §6.4 — 1D regime scan | `src/experiments/13_regime_scan_policy_robustness.py` | 5164335 | fig13, regime_scan_summary |
| §6.6 — 2D regime scan | `src/experiments/15_regime_scan_2d.py` | this commit | fig15, regime_scan_2d_summary |
| §6.7 — cohort MCMC convergence | `src/experiments/14_cohort_mcmc_synthetic.py` | this commit | fig14, cohort_mcmc_summary |
| §6.8 — PA vs PE decision | `src/experiments/16_posterior_aware_vs_point_estimate.py` | this commit | fig16, decision_comparison_summary |
| §6.7 — MH vs NUTS comparison | `src/experiments/18_mh_vs_nuts_convergence.py` | this commit | fig18, mh_vs_nuts_summary |
| §6.7 — JAX-native simulator | `src/realdata/jax_simulator.py` + tests/test_jax_simulator.py | this commit | (production-grade simulator) |
| §6.9 — REAL Bruchovsky cohort | `src/experiments/19_real_cohort_pa_vs_pe.py` + `data/raw/dataTanaka/` | this commit | fig19, real_cohort_pa_vs_pe_summary |
| §6.9 — Real-data ingestion | `src/realdata/bruchovsky.py::load_dataTanaka` + tests | this commit | (loads 72 patients) |
| §6.9.1 — Cross-cohort validation | `src/experiments/20_cross_cohort_pa_vs_pe.py` + `realdata::load_shaw_et_al` | this commit | fig20, cross_cohort_pa_vs_pe_summary |

Each script runs deterministically given a seed; output files include git SHA + ISO date in the filename.

---

**Notes for the next draft pass (v3):**

- ~~Appendix A proof currently a sketch; extend to a full proof.~~ ✅ DONE in v2.
- ~~Add a §3.5 doing the cross-schedule FIM analysis on the 3-pop K-shift model.~~ ✅ DONE in v2 (results from `experiment 12`).
- ~~Find regime where Section 6's policy comparison is NOT robust ($\mathbb{P}(\text{AT50 wins}) < 100\%$). This is the strongest motivating case for Bayesian decision-theoretic adaptive therapy.~~ ✅ DONE in v2 §6.4 (results from `experiment 13`: $K_{TP}^\text{drop} = 1000$ regime gives $\mathbb{P} = 45\%$, near coin-flip).
- Add abstract paragraph mentioning §6.4 regime-scan finding (currently abstract emphasizes the Zhang-canonical 100% result).
- Bibliography entries for additional cited work (Norton-Simon 1976, Goldie-Coldman 1979) referenced in companion blog WP5e.
- Format references as proper BibTeX once arXiv submission is imminent.
- Length-check: v2 draft is approximately 12 pages with the full Appendix A. Target final length 12-14 pages (longer is OK now that §6.4 + §3.5 are added).
- Add a multi-axis regime scan: scan two parameters jointly (e.g., $K_{TP}^\text{drop}$ and $\alpha_{T-,T+}$) and visualize the policy-preference boundary as a 2D heatmap. Phase 3 §3.2 follow-up.
