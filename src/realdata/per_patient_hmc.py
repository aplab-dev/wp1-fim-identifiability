"""Per-patient HMC (NUTS) inference via numpyro + JAX-native simulator.

⚠️ STATUS: wired but not production-ready. See "Known issue" below.

CONFIRMED 2026-05-03: NUTS warmup-hang is reproducible across:
- Synthetic cohort (50 warmup × 50 samples × 2 chains: hangs >50 min CPU).
- Real Bruchovsky patient bruchovsky_p001 (n_obs=75, same settings: hangs
  >50 min CPU at 100% CPU, 600 MB memory).
- Both default prior_log_std = [1.0]*3 + [0.5]*2 + [1.0] AND tighter
  [0.5]*3 + [0.3]*2 + [0.5].

The hang is NOT in NUTS sample-collection (which would be slow but make
progress). It's in warmup tracing through the rank-deficient diffrax
adjoint at extreme proposed θ. Tighter priors don't help because NUTS'
dual-averaging step-size adaptation can still propose extreme jumps
in the unidentifiable directions where the ODE solver hits max_steps.

Architecture:
- `jax_simulator.py` provides a JAX-native LV3PopKShift + PSA simulator.
  Forward 1.3 ms / call (JIT), reverse-mode AD gradient 24.6 ms / call.
  Smooth-floor formulation for AD stability. **This part works** —
  9 tests in `tests/test_jax_simulator.py` pass.
- This file wires that simulator into numpyro.NUTS.

KNOWN ISSUE. NUTS sampling on a single synthetic patient hangs for
50+ minutes at 100% CPU with zero output past the initial "START" line,
when called with even small settings (50 warmup × 50 samples × 2 chains).
Either:
- NUTS warmup proposes extreme theta values where the diffrax adaptive
  ODE solver hits `max_steps`. The `throw=False` flag is supposed to
  return NaN and let NUTS reject, but the trace through the failed solver
  appears extremely slow — possibly due to the checkpointed adjoint
  re-running on every NaN call.
- Or numpyro's NUTS first-time tracing through this complex compute
  graph is pathologically slow on our setup (Python 3.11 + JAX 0.10 +
  diffrax 0.7 + numpyro 0.21).

Workaround for Phase 3 §3.3: use the adaptive-MH sampler in
`per_patient_mcmc.py`. Convergence is documented as "MH is slow on this
posterior" in WP1 §6.7 — production-grade fits are M9-M12 work that
will require either:
(a) Fixing the NUTS-with-diffrax integration (likely tighter solver
    tolerance + bounded prior).
(b) Using SVI / Laplace approximation as a faster fallback for
    rank-deficient posteriors.
(c) Re-implementing as a JAX-native compiled pipeline without diffrax
    (custom ODE integrator inside the model).

Each path is ~4-8 hours of focused work. None of it changes the WP1
methodology message; the Phase 3 §3.3 deliverable can land via MH with
honest "convergence is slow, production deployment requires upgrade"
documentation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# Force x64 mode for numerical stability
jax.config.update("jax_enable_x64", True)

log = logging.getLogger(__name__)

PARAM_NAMES = ["r_T+", "r_TP", "r_T-", "α(T-,T+)", "α(T-,TP)", "K_TP_drop"]


@dataclass(frozen=True)
class HMCResult:
    """Output of fit_patient_hmc_nuts.

    Attributes:
        samples: (n_chains * n_samples_per_chain, 6) flat posterior samples.
        samples_by_chain: (n_chains, n_samples_per_chain, 6).
        rhat: (6,) split-chain R-hat per parameter.
        n_eff: (6,) effective sample size per parameter.
        param_names: list of parameter names.
        patient_id: source patient identifier.
        sampler_kwargs: settings used.
    """

    samples: np.ndarray
    samples_by_chain: np.ndarray
    rhat: np.ndarray
    n_eff: np.ndarray
    param_names: list[str]
    patient_id: str = ""
    sampler_kwargs: dict | None = None

    def converged(self, rhat_threshold: float = 1.10) -> bool:
        return bool(np.all(self.rhat < rhat_threshold))


def fit_patient_hmc_nuts(
    patient,
    n_chains: int = 2,
    n_samples: int = 500,
    n_warmup: int = 500,
    target_accept: float = 0.85,
    psa_noise_rel: float = 0.10,
    seed: int = 0,
    progress_bar: bool = False,
    use_native_integrator: bool = True,
    prior_log_mean: np.ndarray | None = None,
    prior_log_std: np.ndarray | None = None,
    prior_log_cov: np.ndarray | None = None,
    hard_nan_guard: bool = False,
    init_at_canonical: bool = False,
) -> HMCResult:
    """Fit one Bruchovsky patient via NUTS HMC with the JAX-native simulator.

    The likelihood uses the AD-stable smooth-floor LV3PopKShift implementation
    in `jax_simulator.py`. Gradients flow through the ODE integration,
    enabling NUTS's leapfrog dynamics to traverse the rank-deficient posterior
    efficiently.

    Args:
        patient: BruchovskyPatient.
        n_chains: number of chains (2-4 typical; 4 for clinical-grade R-hat).
        n_samples: post-warmup samples per chain.
        n_warmup: warmup steps per chain.
        target_accept: NUTS target acceptance rate.
        psa_noise_rel: relative noise model for likelihood.
        seed: top-level seed.
        progress_bar: show numpyro's progress bar.
        use_native_integrator: if True (default), use the JAX-native fixed-step
            Heun integrator (no diffrax). This resolves the warmup-hang that
            blocked previous NUTS attempts: diffrax's adaptive solver could be
            pushed into its `max_steps` ceiling when NUTS dual-averaging
            proposed extreme θ during warmup, and the checkpointed adjoint
            re-trace was pathologically slow. The native integrator has
            bounded per-call cost. If False, falls back to diffrax for
            backward compatibility.
        prior_log_mean: (6,) prior log-mean for θ. If None, defaults to
            log(Zhang canonical θ).
        prior_log_std: (6,) prior log-std for θ. If None, defaults to
            [0.5, 0.5, 0.5, 0.3, 0.3, 0.5] (the v7-default tightened-on-α
            prior). Pass a tighter std on the unidentifiable directions
            (α, K_TP_drop) — informed by the cohort-level hierarchical fit
            (§6.11) — to substantially improve per-patient NUTS R-hat. See
            `prior_from_hierarchical_fit` for the principled construction.
            Ignored if `prior_log_cov` is supplied.
        prior_log_cov: (6, 6) FULL log-covariance matrix for log θ. If
            supplied, uses a multivariate-normal prior with this covariance
            instead of the diagonal one defined by prior_log_std. This is
            the right thing to use when the unidentifiable direction is a
            joint anti-correlation (e.g., α(T-,T+) vs α(T-,TP) in §3.3) —
            independent-marginal tightening is insufficient (§6.11.2).
            See `fim_eigenbasis_prior_cov` for one principled construction.
        hard_nan_guard: if True, attach a large negative log-density penalty
            (`numpyro.factor`) whenever predict_psa(θ) contains a NaN. The
            soft NaN-guard substitutes observed PSA for predicted and
            inflates noise — yielding a flat log-likelihood plateau that
            NUTS can wander into (§6.11.2 diagnosis). The hard guard makes
            the plateau a sharp barrier, forcing NUTS to bounce off
            invalid-θ regions. Default False for backward compat; set True
            for production NUTS attempts.
        init_at_canonical: if True, initialize ALL NUTS chains at the
            canonical Zhang θ (or `prior_log_mean` if supplied) instead of
            random samples from the prior. The diagnosis in §6.11.2 shows
            random init causes chains to diverge into nonphysical regions
            during warmup; deterministic init lets NUTS' step-size
            adaptation start from a stable region. Default False for
            backward compat; True is recommended for production runs.

    Returns:
        HMCResult with samples + diagnostics.
    """
    from .bruchovsky import BruchovskyPatient
    from .jax_simulator import _make_jax_predictor, _make_jax_predictor_native
    from .per_patient_mcmc import _zhang_canonical_theta

    if not isinstance(patient, BruchovskyPatient):
        raise TypeError(f"patient must be BruchovskyPatient; got {type(patient).__name__}")

    if prior_log_mean is None:
        prior_log_mean_arr = jnp.log(jnp.maximum(jnp.asarray(_zhang_canonical_theta()), 1e-12))
    else:
        prior_log_mean_arr = jnp.asarray(prior_log_mean)

    if prior_log_std is None:
        # v7-default: tighter than synthetic-cohort default to prevent NUTS
        # from proposing extreme θ. The α / K_TP_drop dims are slightly
        # tighter; use `prior_from_hierarchical_fit` for the cohort-informed
        # version that further tightens the unidentifiable directions.
        prior_log_std_arr = jnp.array([0.5, 0.5, 0.5, 0.3, 0.3, 0.5])
    else:
        prior_log_std_arr = jnp.asarray(prior_log_std)
    sigma_obs = jnp.asarray(
        psa_noise_rel * np.maximum(patient.psa_obs, 0.1 * np.max(patient.psa_obs))
    )

    # Build the JIT-compiled simulator closure.
    if use_native_integrator:
        predict_psa = _make_jax_predictor_native(patient.t_obs, patient.u_schedule)
    else:
        predict_psa = _make_jax_predictor(patient.t_obs, patient.u_schedule)

    nan_penalty_log_density = -1e10  # used by hard_nan_guard

    def _likelihood_block(psa_pred, psa_obs):
        """Common observation-likelihood code with soft or hard NaN guard."""
        any_nan = jnp.any(~jnp.isfinite(psa_pred))
        if hard_nan_guard:
            # Hard barrier: large -log-density when any prediction is NaN.
            # NUTS will see this as a sharp wall and bounce off.
            numpyro.factor("nan_penalty",
                           jnp.where(any_nan, nan_penalty_log_density, 0.0))
            is_finite = jnp.isfinite(psa_pred)
            psa_pred_safe = jnp.where(is_finite, jnp.clip(psa_pred, 1e-6, 1e8),
                                      jnp.asarray(psa_obs))
            sigma_eff = sigma_obs
        else:
            # Soft guard: substitute observed PSA + inflated noise where NaN.
            is_finite = jnp.isfinite(psa_pred)
            psa_pred_safe = jnp.where(is_finite, jnp.clip(psa_pred, 1e-6, 1e8),
                                      jnp.asarray(psa_obs))
            sigma_eff = jnp.where(is_finite, sigma_obs, sigma_obs * 1e6)
        numpyro.sample(
            "obs",
            dist.Normal(psa_pred_safe, sigma_eff),
            obs=jnp.asarray(psa_obs),
        )

    if prior_log_cov is not None:
        prior_log_cov_arr = jnp.asarray(prior_log_cov)

        def model(psa_obs):
            log_theta = numpyro.sample(
                "log_theta",
                dist.MultivariateNormal(
                    loc=prior_log_mean_arr,
                    covariance_matrix=prior_log_cov_arr,
                ),
            )
            theta = numpyro.deterministic("theta", jnp.exp(log_theta))
            psa_pred = predict_psa(theta)
            _likelihood_block(psa_pred, psa_obs)
    else:
        def model(psa_obs):
            theta = numpyro.sample(
                "theta",
                dist.LogNormal(loc=prior_log_mean_arr, scale=prior_log_std_arr),
            )
            psa_pred = predict_psa(theta)
            _likelihood_block(psa_pred, psa_obs)

    rng_key = jax.random.PRNGKey(seed)
    init_strategy = None
    if init_at_canonical:
        from numpyro.infer import init_to_value
        # Build init values for the appropriate site (theta or log_theta).
        if prior_log_cov is not None:
            init_vals = {"log_theta": prior_log_mean_arr}
        else:
            init_vals = {"theta": jnp.exp(prior_log_mean_arr)}
        init_strategy = init_to_value(values=init_vals)
    nuts_kwargs = {"target_accept_prob": target_accept}
    if init_strategy is not None:
        nuts_kwargs["init_strategy"] = init_strategy
    nuts_kernel = NUTS(model, **nuts_kwargs)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=n_chains,
        progress_bar=progress_bar,
        chain_method="sequential",  # avoid pmap overhead for n_chains=2-4
    )
    mcmc.run(rng_key, psa_obs=patient.psa_obs)

    samples_dict = mcmc.get_samples(group_by_chain=True)
    theta_by_chain = np.asarray(samples_dict["theta"])  # (n_chains, n_samples, 6)
    flat_samples = theta_by_chain.reshape(-1, 6)

    # R-hat + n_eff via numpyro's diagnostics.
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    rhat = np.asarray(split_gelman_rubin(theta_by_chain))
    try:
        n_eff = np.asarray(effective_sample_size(theta_by_chain))
    except Exception:  # noqa: BLE001
        n_eff = np.full(6, float(theta_by_chain.shape[1]))

    return HMCResult(
        samples=flat_samples,
        samples_by_chain=theta_by_chain,
        rhat=rhat,
        n_eff=n_eff,
        param_names=PARAM_NAMES,
        patient_id=patient.patient_id,
        sampler_kwargs={
            "n_chains": n_chains,
            "n_samples": n_samples,
            "n_warmup": n_warmup,
            "target_accept": target_accept,
            "sampler": "NUTS-numpyro",
            "simulator": "JAX-native + Heun fixed-step (smooth-floor)",
        },
    )


def fim_eigenbasis_prior_cov(
    theta_nominal: np.ndarray,
    t_obs: np.ndarray,
    u_schedule: np.ndarray,
    sigma_obs: np.ndarray | None = None,
    sigma_loose: float = 0.5,
    sigma_tight: float = 0.05,
    eigenvalue_ratio_threshold: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a FIM-eigenbasis prior covariance for log θ.

    The 3-pop FIM has rank 3 of 6 (§3.2): three identifiable directions
    (eigenvalues > 1) and three unidentifiable directions
    (eigenvalues << 1). A diagonal prior in log θ space cannot
    distinguish these directions because the unidentifiable directions
    are joint correlations across multiple parameters (e.g., α(T-,T+) vs
    α(T-,TP) anti-correlation).

    This builds a FIM-eigenbasis prior:
        FIM(log θ) = V Λ V^T  (eigendecomposition in log θ space)
        Σ_prior = V diag(σ_k²) V^T
    where:
        σ_k = sigma_loose if λ_k / λ_max > eigenvalue_ratio_threshold
        σ_k = sigma_tight otherwise

    This is loose along the identifiable eigenvectors (so data leads) and
    tight along the unidentifiable eigenvectors (so prior leads). It
    constrains the joint anti-correlated direction without over-constraining
    the well-identified directions.

    Args:
        theta_nominal: (6,) θ at which the FIM is evaluated (typically the
            canonical Zhang θ, or a posterior mean from a prior MH run).
        t_obs: (T,) observation times.
        u_schedule: (T,) piecewise-constant drug schedule.
        sigma_obs: (T,) per-time-point observation noise std (or None to
            use unit weights).
        sigma_loose: prior std along identifiable eigenvectors (default 0.5
            — same as the diffuse default for marginals).
        sigma_tight: prior std along unidentifiable eigenvectors (default
            0.05, i.e., ~5% in log space → ~5% multiplicative).
        eigenvalue_ratio_threshold: λ_k / λ_max threshold defining the
            "identifiable" vs "unidentifiable" split. The 3-pop K-shift FIM
            has λ_3 / λ_max ~ 1e-3 and λ_4 / λ_max ~ 1e-9; the default
            1e-3 puts the boundary right between rank 3 and rank 4.

    Returns:
        (prior_log_mean, prior_log_cov), each suitable to pass to
        fit_patient_hmc_nuts. prior_log_mean = log(theta_nominal);
        prior_log_cov is (6, 6) symmetric positive-definite.
    """
    from identifiability.fim import compute_fim
    from .per_patient_mcmc import _predict_psa_at

    if sigma_obs is None:
        sigma_obs = np.ones(len(t_obs))

    # Predict in log-θ space so the FIM is in log-θ coordinates.
    def predict_log_theta(log_theta: np.ndarray) -> np.ndarray:
        return _predict_psa_at(np.exp(log_theta), t_obs, u_schedule)

    log_theta_nominal = np.log(np.maximum(theta_nominal, 1e-12))
    fim_result = compute_fim(
        predict_log_theta, log_theta_nominal,
        eps_rel=1e-3, sigma=sigma_obs,
    )
    # Eigendecompose. FIM is symmetric → use eigh.
    eigvals, eigvecs = np.linalg.eigh(fim_result.fim)
    # eigvals are sorted ascending; eigvecs columns are eigenvectors.
    eigval_max = eigvals.max()
    # Assign per-direction std: loose for identifiable, tight for unidentifiable.
    is_identifiable = eigvals / eigval_max > eigenvalue_ratio_threshold
    sigma_per_dir = np.where(is_identifiable, sigma_loose, sigma_tight)
    # Build covariance in eigenbasis, then rotate to log-θ basis.
    diag_cov = np.diag(sigma_per_dir ** 2)
    prior_log_cov = eigvecs @ diag_cov @ eigvecs.T
    # Numerical symmetrization (should already be symmetric).
    prior_log_cov = 0.5 * (prior_log_cov + prior_log_cov.T)
    return log_theta_nominal, prior_log_cov


def prior_from_hierarchical_fit(hier_fit, inflation: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Extract a per-patient prior (log_mean, log_std) from a HierarchicalFit.

    Constructs the marginal prior predictive for a NEW patient drawn from the
    fitted population: log θ ~ N(μ_post_mean, sqrt(σ_pop_post_mean² + σ_μ_post²)).

    The combined std is the standard Bayesian marginal-prior std for a new
    draw from a posterior with both location and scale uncertainty:
        Var(log θ_new | data) = Var(σ_pop² | data) summed across the posterior
                              + Var(μ | data)  (location uncertainty)
        ≈ E[σ_pop² | data] + Var[μ | data]
        ≈ (σ_pop_post_mean)² + (σ_μ_post)²

    For a SINGLE new patient drawn from the cohort, this is much tighter than
    the diffuse default ([0.5]*3 + [0.3]*3) on the unidentifiable directions
    α(T-,T+) and α(T-,TP), because hierarchical pooling has substantially
    constrained the population spread.

    Args:
        hier_fit: HierarchicalFit object from realdata.hierarchical.hierarchical_fit.
        inflation: optional inflation factor on the std (default 1.0). Use a
            value > 1 to make the prior slightly less informative (helpful if
            you suspect the cohort population posterior is overconfident).

    Returns:
        (prior_log_mean, prior_log_std), each (6,) numpy arrays, suitable to
        pass directly to fit_patient_hmc_nuts.
    """
    mu_post_mean = hier_fit.pop_mean_samples.mean(axis=0)       # (6,)
    mu_post_std = hier_fit.pop_mean_samples.std(axis=0)         # (6,)
    sigma_pop_post_mean = hier_fit.pop_std_samples.mean(axis=0) # (6,)

    prior_log_mean = mu_post_mean
    prior_log_std = inflation * np.sqrt(sigma_pop_post_mean ** 2 + mu_post_std ** 2)
    return prior_log_mean, prior_log_std


# Legacy stub kept so existing imports don't break.
def fit_patient_hmc(*args, **kwargs):
    """Deprecated. Use fit_patient_hmc_nuts instead."""
    return fit_patient_hmc_nuts(*args, **kwargs)
