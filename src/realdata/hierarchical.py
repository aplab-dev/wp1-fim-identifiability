"""Hierarchical Bayesian fit pooling across the Bruchovsky cohort.

Closes WP1 §7.2 limitation #1 ("single-patient single-realization FIM is a
worst-case-per-patient bound, not a cohort-fitting bound").

Approach: take the per-patient MCMC posteriors from experiment 19 as
summary likelihoods (Gaussian on log θ), then fit a hierarchical model:

    μ_k ~ N(prior_mean_k, prior_std_k)         [population mean per param]
    σ_k ~ HalfNormal(prior_scale_k)            [population SD per param]
    log θ^(i)_k ~ N(μ_k, σ_k)                  [per-patient draw]
    log θ̂^(i)_k ~ N(log θ^(i)_k, fit_std^(i)_k)  [observed posterior summary]

This pools information across patients to constrain the population-level
distribution. Even though each patient's posterior is rank-deficient,
pooling 72 of them constrains the combined posterior on (μ, σ) substantially.

The model is closed-form Gaussian — NUTS/HMC works fine on it via numpyro
because no diffrax simulator is in the inner loop. This is what unblocks
the production-grade Bayesian fit that the per-patient HMC couldn't deliver.

Output (when run as experiment 22):
- Population-level posterior on (μ_k, σ_k) for each of 6 parameters.
- Shrunk per-patient posteriors that pool across cohort.
- Comparison: pooled vs unpooled per-patient posterior widths.

Usage:
    from realdata import generate_synthetic_cohort
    from realdata.hierarchical import (
        per_patient_summaries, hierarchical_fit, compare_pooled_vs_unpooled,
    )

    # Step 1: collect per-patient posterior summaries (Gaussian fit to each MCMC chain).
    summaries = per_patient_summaries(cohort, mcmc_results=mcmc_per_patient)

    # Step 2: hierarchical fit on the summaries.
    h = hierarchical_fit(summaries, n_chains=4, n_samples=1000)

    # Step 3: shrunk per-patient posteriors.
    shrunk = h.shrunk_per_patient(summaries)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

PARAM_NAMES = ["r_T+", "r_TP", "r_T-", "α(T-,T+)", "α(T-,TP)", "K_TP_drop"]


@dataclass(frozen=True)
class PatientSummary:
    """Gaussian summary of one patient's per-patient MCMC posterior.

    Used as the likelihood input to the hierarchical fit.
    """

    patient_id: str
    log_theta_mean: np.ndarray  # (6,) — mean of log θ per parameter
    log_theta_std: np.ndarray   # (6,) — std of log θ per parameter (per-patient observed sigma)


@dataclass(frozen=True)
class HierarchicalFit:
    """Result of fit_hierarchical."""

    pop_mean_samples: np.ndarray   # (n_chain*n_samples, 6) — μ_k posterior samples
    pop_std_samples: np.ndarray    # (n_chain*n_samples, 6) — σ_k posterior samples
    rhat_pop_mean: np.ndarray      # (6,) per-parameter R-hat for μ_k
    rhat_pop_std: np.ndarray       # (6,) per-parameter R-hat for σ_k
    n_patients: int
    param_names: list[str] = field(default_factory=lambda: PARAM_NAMES.copy())

    def converged(self, rhat_threshold: float = 1.10) -> bool:
        return bool(np.all(self.rhat_pop_mean < rhat_threshold)
                    and np.all(self.rhat_pop_std < rhat_threshold))


def per_patient_summaries(mcmc_results: list, patient_ids: list[str] | None = None) -> list[PatientSummary]:
    """Convert per-patient MCMC posterior samples into Gaussian summaries.

    Args:
        mcmc_results: list of per-patient sample arrays (n_samples, 6) or
            full MCMCResult-like objects with .flat_samples().
        patient_ids: optional list of patient IDs (defaults to "p_{i}").

    Returns:
        List of PatientSummary objects.
    """
    summaries = []
    for i, result in enumerate(mcmc_results):
        if hasattr(result, "flat_samples"):
            samples = result.flat_samples()
            pid = getattr(result, "patient_id", None) or f"p_{i:03d}"
        else:
            samples = np.asarray(result)
            pid = patient_ids[i] if patient_ids else f"p_{i:03d}"
        if samples.ndim != 2 or samples.shape[1] != 6:
            log.warning(f"  patient {pid}: unexpected shape {samples.shape}, skipping")
            continue
        # Work in log space.
        log_samples = np.log(np.maximum(samples, 1e-12))
        summaries.append(PatientSummary(
            patient_id=pid,
            log_theta_mean=log_samples.mean(axis=0),
            log_theta_std=log_samples.std(axis=0) + 1e-3,  # tiny floor for numerical stability
        ))
    return summaries


def hierarchical_fit(
    summaries: list[PatientSummary],
    prior_pop_mean: np.ndarray | None = None,
    prior_pop_mean_std: np.ndarray | None = None,
    prior_pop_scale_std: np.ndarray | None = None,
    n_chains: int = 4,
    n_samples: int = 1000,
    n_warmup: int = 500,
    seed: int = 0,
    progress_bar: bool = False,
    centered: bool = False,
    target_accept_prob: float = 0.95,
) -> HierarchicalFit:
    """Run NUTS HMC on the hierarchical Gaussian model.

    Centered parameterization (centered=True):
        μ[k] ~ N(prior_pop_mean[k], prior_pop_mean_std[k])
        σ_pop[k] ~ HalfNormal(prior_pop_scale_std[k])
        true_log_θ[i, k] ~ N(μ[k], σ_pop[k])
        log_θ_obs[i, k] ~ N(true_log_θ[i, k], σ_obs[i, k])

    Non-centered parameterization (centered=False — DEFAULT):
        μ[k] ~ N(prior_pop_mean[k], prior_pop_mean_std[k])
        σ_pop[k] ~ HalfNormal(prior_pop_scale_std[k])
        z[i, k] ~ N(0, 1)                      # standardized per-patient draw
        true_log_θ[i, k] := μ[k] + σ_pop[k] * z[i, k]    # deterministic
        log_θ_obs[i, k] ~ N(true_log_θ[i, k], σ_obs[i, k])

    The non-centered version eliminates Neal's funnel between σ_pop and the
    per-patient latents — NUTS now samples z (unit-normal everywhere) instead
    of true_log_θ (whose marginal width depends on the realized σ_pop). This
    is the standard fix for hierarchical-model σ_pop convergence problems and
    is recommended whenever per-patient summary likelihoods are sharper than
    the population spread (which is our regime).

    The model is closed-form Gaussian; numpyro NUTS handles it directly with
    no diffrax involvement (so no NUTS warmup hang).

    Args:
        summaries: list of PatientSummary objects (length n_patients).
        prior_pop_mean: (6,) prior mean for μ_k. Default = 0 (in log space).
        prior_pop_mean_std: (6,) prior std for μ_k. Default = 5 (very loose).
        prior_pop_scale_std: (6,) HalfNormal scale for σ_pop_k. Default = 2.
        n_chains, n_samples, n_warmup, seed: NUTS settings.
        progress_bar: show numpyro's progress bar.
        centered: if True, use the centered parameterization (the v1
            implementation; produced R-hat = 1.47 on σ_pop for α(T-,T+) on the
            71-patient Bruchovsky run). If False (default), use the non-centered
            parameterization which removes Neal's funnel and yields better
            sampler geometry. Provided for backwards-compatibility + replication.
        target_accept_prob: NUTS target acceptance. Default raised from 0.85
            to 0.95 to handle the unidentifiable-direction marginal more
            carefully (smaller step size in those directions).

    Returns:
        HierarchicalFit with population-level samples + R-hat diagnostics.
    """
    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.diagnostics import split_gelman_rubin
        from numpyro.infer import MCMC, NUTS
    except ImportError as e:
        raise ImportError(
            "fit_hierarchical requires numpyro+jax. Install via: uv pip install numpyro jax"
        ) from e

    jax.config.update("jax_enable_x64", True)

    if not summaries:
        raise ValueError("no patient summaries provided")

    n_patients = len(summaries)
    log_theta_obs = np.array([s.log_theta_mean for s in summaries])  # (N, 6)
    sigma_obs = np.array([s.log_theta_std for s in summaries])       # (N, 6)

    if prior_pop_mean is None:
        prior_pop_mean = np.zeros(6)
    if prior_pop_mean_std is None:
        prior_pop_mean_std = np.full(6, 5.0)
    if prior_pop_scale_std is None:
        prior_pop_scale_std = np.full(6, 2.0)

    log_theta_obs_j = jnp.asarray(log_theta_obs)
    sigma_obs_j = jnp.asarray(sigma_obs)
    prior_pop_mean_j = jnp.asarray(prior_pop_mean)
    prior_pop_mean_std_j = jnp.asarray(prior_pop_mean_std)
    prior_pop_scale_std_j = jnp.asarray(prior_pop_scale_std)

    if centered:
        def model():
            mu = numpyro.sample("mu", dist.Normal(prior_pop_mean_j, prior_pop_mean_std_j))
            sigma = numpyro.sample("sigma", dist.HalfNormal(prior_pop_scale_std_j))

            # True (latent) log-θ per patient
            with numpyro.plate("patients", n_patients):
                true_log_theta = numpyro.sample(
                    "true_log_theta",
                    dist.Normal(mu, sigma).expand([n_patients, 6]).to_event(1),
                )

            # Observed Gaussian summary likelihood
            numpyro.sample(
                "obs",
                dist.Normal(true_log_theta, sigma_obs_j).to_event(1),
                obs=log_theta_obs_j,
            )
    else:
        def model():
            # Non-centered: sample standardized z, build true_log_theta as deterministic.
            mu = numpyro.sample("mu", dist.Normal(prior_pop_mean_j, prior_pop_mean_std_j))
            sigma = numpyro.sample("sigma", dist.HalfNormal(prior_pop_scale_std_j))

            with numpyro.plate("patients", n_patients):
                z = numpyro.sample(
                    "z",
                    dist.Normal(jnp.zeros((n_patients, 6)),
                                jnp.ones((n_patients, 6))).to_event(1),
                )

            # true_log_theta is deterministic given (mu, sigma, z) — broadcast over plate
            true_log_theta = mu[None, :] + sigma[None, :] * z  # (N, 6)
            numpyro.deterministic("true_log_theta", true_log_theta)

            # Observed Gaussian summary likelihood
            numpyro.sample(
                "obs",
                dist.Normal(true_log_theta, sigma_obs_j).to_event(1),
                obs=log_theta_obs_j,
            )

    rng_key = jax.random.PRNGKey(seed)
    nuts_kernel = NUTS(model, target_accept_prob=target_accept_prob)
    mcmc = MCMC(
        nuts_kernel, num_warmup=n_warmup, num_samples=n_samples,
        num_chains=n_chains, progress_bar=progress_bar, chain_method="sequential",
    )
    mcmc.run(rng_key)

    samples = mcmc.get_samples(group_by_chain=True)
    mu_samples = np.asarray(samples["mu"])       # (n_chains, n_samples, 6)
    sigma_samples = np.asarray(samples["sigma"]) # (n_chains, n_samples, 6)

    rhat_mu = np.asarray(split_gelman_rubin(mu_samples))
    rhat_sigma = np.asarray(split_gelman_rubin(sigma_samples))

    return HierarchicalFit(
        pop_mean_samples=mu_samples.reshape(-1, 6),
        pop_std_samples=sigma_samples.reshape(-1, 6),
        rhat_pop_mean=rhat_mu,
        rhat_pop_std=rhat_sigma,
        n_patients=n_patients,
    )


def compare_pooled_vs_unpooled(
    summaries: list[PatientSummary], h: HierarchicalFit,
) -> dict:
    """Compute the per-parameter shrinkage from unpooled to pooled posterior std.

    For each patient × parameter:
        unpooled std = the per-patient MCMC posterior std on log θ.
        pooled std   = the posterior-predictive std for that patient under
                       the hierarchical model (≈ the std of the conditional
                       distribution of log θ given the population posterior).

    The pooled std is typically tighter than the unpooled std for the
    unidentifiable directions, by approximately the factor
        sqrt(σ_pop² / (σ_pop² + σ_obs²))
    weighted by the population-level posterior on σ_pop.

    Returns:
        Dict with per-parameter shrinkage statistics.
    """
    n_patients = len(summaries)
    sigma_obs = np.array([s.log_theta_std for s in summaries])  # (N, 6)
    sigma_pop_post_mean = h.pop_std_samples.mean(axis=0)        # (6,)

    # Pooled per-patient std under the hierarchical posterior, integrating
    # over σ_pop ≈ E_σ[1 / (1/σ_pop² + 1/σ_obs²)^0.5].
    # Simple approximation using point estimate of σ_pop:
    pooled_std = np.sqrt(1.0 / (1.0 / sigma_pop_post_mean[None, :] ** 2
                                + 1.0 / sigma_obs ** 2))

    shrinkage = pooled_std / sigma_obs  # < 1.0 means pooling tightened the per-patient posterior

    return {
        "param_names": PARAM_NAMES,
        "n_patients": n_patients,
        "sigma_obs_per_patient_per_param": sigma_obs.tolist(),
        "sigma_pop_posterior_mean": sigma_pop_post_mean.tolist(),
        "pooled_std_per_patient_per_param": pooled_std.tolist(),
        "shrinkage_factor_per_patient_per_param": shrinkage.tolist(),
        "median_shrinkage_per_param": np.median(shrinkage, axis=0).tolist(),
        "mean_shrinkage_per_param": shrinkage.mean(axis=0).tolist(),
    }
