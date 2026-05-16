"""Per-patient Bayesian inference via adaptive Metropolis-Hastings MCMC.

Implements the per-patient layer of Phase 3 §3.3:
- Likelihood: Gaussian residuals on PSA observations with 10% relative noise.
- Prior: weak Gaussian centered at the population-mean theta (Zhang canonical),
  with regime-appropriate scale; positivity constraints.
- Sampler: adaptive component-wise Metropolis-Hastings (~70 lines, no
  external Bayesian-inference dep). Multiple chains for R-hat diagnostics.

Why MH and not HMC? Per the v1 §5 finding, the FIM-Gaussian dramatically
underestimates posterior uncertainty in unidentifiable directions. We
need a sampler that handles wide unidentifiable directions without
gradient information, since the gradient itself is near-zero in those
directions. Adaptive MH with appropriate proposal-scaling does this
robustly in d=6.

Returns full posterior samples + R-hat per parameter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from simulators.lv_3pop_kshift import LV3PopKShift, LV3PopParams
from simulators.psa_dynamics import PSAParams, psa_steady_state
from zhang2017 import ZHANG_CANONICAL_X0, zhang_canonical_lv_params

from .bruchovsky import BruchovskyPatient

log = logging.getLogger(__name__)


PARAM_NAMES = ["r_T+", "r_TP", "r_T-", "α(T-,T+)", "α(T-,TP)", "K_TP_drop"]


@dataclass(frozen=True)
class MCMCResult:
    """Output of fit_patient_mcmc.

    Attributes:
        samples: (n_chains, n_samples_per_chain, 6) post-burnin thinned samples.
        param_names: length-6 names (consistent with simulator subset).
        rhat: (6,) R-hat split-chain convergence diagnostic per parameter. <1.05 = converged.
        accept_rates: (n_chains, 6) per-component acceptance rates at end of burn-in.
        log_posterior_traces: (n_chains, n_total_steps) log-posterior trace per chain.
        proposal_scales_final: (n_chains, 6) final per-component proposal scales.
        theta_init: starting theta for each chain.
        patient_id: source patient identifier.
        n_total_steps: total MCMC steps including burn-in.
        burn_in: burn-in steps.
        thin: thinning interval.
    """

    samples: np.ndarray
    param_names: list[str]
    rhat: np.ndarray
    accept_rates: np.ndarray
    log_posterior_traces: np.ndarray
    proposal_scales_final: np.ndarray
    theta_init: np.ndarray
    patient_id: str = ""
    n_total_steps: int = 0
    burn_in: int = 0
    thin: int = 1

    @property
    def n_chains(self) -> int:
        return self.samples.shape[0]

    @property
    def n_samples_per_chain(self) -> int:
        return self.samples.shape[1]

    def flat_samples(self) -> np.ndarray:
        """Concatenate all chains: (n_chains * n_samples_per_chain, 6)."""
        return self.samples.reshape(-1, self.samples.shape[-1])

    def converged(self, rhat_threshold: float = 1.10) -> bool:
        """All parameters have R-hat < threshold (default 1.10 for clinical-grade rigor)."""
        return bool(np.all(self.rhat < rhat_threshold))


def rhat_split(samples: np.ndarray) -> np.ndarray:
    """Split-chain R-hat (Gelman-Rubin) per parameter.

    Args:
        samples: (n_chains, n_samples, n_params).

    Returns:
        (n_params,) R-hat. Values close to 1.0 indicate convergence.
    """
    n_chains, n_samples, n_params = samples.shape
    if n_chains < 2 or n_samples < 4:
        return np.full(n_params, np.inf)
    # Split each chain in half.
    half = n_samples // 2
    s = np.concatenate([samples[:, :half, :], samples[:, half:2 * half, :]], axis=0)
    M, N = s.shape[0], s.shape[1]
    chain_means = s.mean(axis=1)  # (M, n_params)
    chain_vars = s.var(axis=1, ddof=1)  # (M, n_params)
    overall_mean = chain_means.mean(axis=0)
    B = N * np.sum((chain_means - overall_mean) ** 2, axis=0) / (M - 1)
    W = chain_vars.mean(axis=0)
    var_hat = (N - 1) / N * W + B / N
    rhat = np.sqrt(var_hat / np.maximum(W, 1e-30))
    return rhat


def _build_lv_params(theta: np.ndarray) -> LV3PopParams:
    canon = zhang_canonical_lv_params()
    alpha = canon.alpha.copy()
    alpha[2, 0] = max(theta[3], 0.01)
    alpha[2, 1] = max(theta[4], 0.01)
    return LV3PopParams(
        r_Tplus=max(theta[0], 1e-6), r_TP=max(theta[1], 1e-6), r_Tminus=max(theta[2], 1e-6),
        K_Tminus=canon.K_Tminus, K_TP_max=canon.K_TP_max,
        K_TP_drop=max(min(theta[5], canon.K_TP_max - 1), 1.0),
        mu_max=canon.mu_max, mu_drop=canon.mu_drop,
        alpha=alpha,
    )


def _predict_psa_at(theta: np.ndarray, t_obs: np.ndarray, u_schedule: np.ndarray) -> np.ndarray | None:
    """Simulate PSA under a per-measurement piecewise-constant schedule. Returns PSA at t_obs."""
    sim = LV3PopKShift(_build_lv_params(theta))
    psa_params = PSAParams()

    def rhs(t, y):
        x = y[:3]
        psa = y[3]
        idx = int(np.searchsorted(t_obs, t, side="right") - 1)
        idx = max(0, min(idx, len(u_schedule) - 1))
        u = float(u_schedule[idx])
        dx = sim.dynamics(t, x, u)
        dpsa = psa_params.rho * float(np.sum(x)) - psa_params.phi * psa
        return np.concatenate([dx, [dpsa]])

    psa0 = psa_steady_state(float(np.sum(ZHANG_CANONICAL_X0)), psa_params)
    y0 = np.array([*ZHANG_CANONICAL_X0, psa0])
    for method in ("LSODA", "BDF"):
        try:
            sol = solve_ivp(rhs, t_span=(t_obs[0], t_obs[-1]), y0=y0, t_eval=t_obs,
                            method=method, rtol=1e-7, atol=1e-3)
            if sol.success:
                return sol.y[3]
        except Exception:  # noqa: BLE001
            continue
    return None


def _is_physically_valid(theta: np.ndarray) -> bool:
    return (
        theta[0] > 0 and theta[1] > 0 and theta[2] > 0
        and theta[3] >= 0 and theta[4] >= 0
        and 0 < theta[5] < 9999.0
    )


def _zhang_canonical_theta() -> np.ndarray:
    canon = zhang_canonical_lv_params()
    return np.array([
        canon.r_Tplus, canon.r_TP, canon.r_Tminus,
        float(canon.alpha[2, 0]), float(canon.alpha[2, 1]),
        canon.K_TP_drop,
    ])


def _log_posterior(
    theta: np.ndarray,
    patient: BruchovskyPatient,
    sigma: np.ndarray,
    prior_mean: np.ndarray,
    prior_log_std: np.ndarray,
) -> float:
    """Log posterior = log likelihood + log prior. Improper if outside physical bounds."""
    if not _is_physically_valid(theta):
        return -np.inf
    psa_pred = _predict_psa_at(theta, patient.t_obs, patient.u_schedule)
    if psa_pred is None:
        return -np.inf
    residuals = (patient.psa_obs - psa_pred) / sigma
    log_lik = -0.5 * float(np.sum(residuals ** 2))
    # Weak Gaussian prior in log space (positivity already enforced).
    log_theta = np.log(np.maximum(theta, 1e-12))
    log_mean = np.log(np.maximum(prior_mean, 1e-12))
    log_prior = -0.5 * float(np.sum(((log_theta - log_mean) / prior_log_std) ** 2))
    return log_lik + log_prior


def _adaptive_mh_chain(
    log_post_fn,
    theta_init: np.ndarray,
    n_steps: int,
    burn_in: int,
    thin: int,
    rng: np.random.Generator,
    proposal_scale_init: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one adaptive MH chain. Returns (samples, accept_rate_per_dim, log_post_trace, final_scale)."""
    n_dim = len(theta_init)
    theta = theta_init.copy()
    log_p = log_post_fn(theta)
    if not np.isfinite(log_p):
        # Try a few random restarts within ~10% of initial
        for _ in range(50):
            jitter = theta_init * np.exp(rng.normal(0, 0.1, size=n_dim))
            log_p_try = log_post_fn(jitter)
            if np.isfinite(log_p_try):
                theta = jitter
                log_p = log_p_try
                break
        if not np.isfinite(log_p):
            raise ValueError("Could not find finite-log-posterior starting point near theta_init")

    samples = []
    proposal_scale = proposal_scale_init.copy()
    accept_counter = np.zeros(n_dim)
    propose_counter = np.zeros(n_dim)
    log_post_trace = np.zeros(n_steps)

    for step in range(n_steps):
        dim = step % n_dim
        delta = rng.normal() * proposal_scale[dim]
        theta_proposed = theta.copy()
        theta_proposed[dim] = theta[dim] + delta
        log_p_proposed = log_post_fn(theta_proposed)
        log_alpha = log_p_proposed - log_p
        propose_counter[dim] += 1
        if np.log(rng.uniform()) < log_alpha:
            theta = theta_proposed
            log_p = log_p_proposed
            accept_counter[dim] += 1
        log_post_trace[step] = log_p

        # Adapt proposal scale during burn-in.
        if step < burn_in and step > 0 and step % (10 * n_dim) == 0:
            for d in range(n_dim):
                if propose_counter[d] >= 5:
                    rate = accept_counter[d] / propose_counter[d]
                    if rate < 0.20:
                        proposal_scale[d] *= 0.85
                    elif rate > 0.45:
                        proposal_scale[d] *= 1.15
                    accept_counter[d] = 0
                    propose_counter[d] = 0

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(theta.copy())

    final_accept_rates = np.divide(
        accept_counter, np.maximum(propose_counter, 1),
        out=np.zeros_like(accept_counter), where=propose_counter > 0,
    )
    return np.array(samples), final_accept_rates, log_post_trace, proposal_scale


def fit_patient_mcmc(
    patient: BruchovskyPatient,
    n_chains: int = 3,
    n_steps: int = 3000,
    burn_in: int = 1000,
    thin: int = 4,
    psa_noise_rel: float = 0.10,
    prior_log_std: np.ndarray | None = None,
    theta_init_fn=None,
    seed: int = 0,
    verbose: bool = False,
) -> MCMCResult:
    """Run multi-chain adaptive MH MCMC on one Bruchovsky patient.

    Args:
        patient: BruchovskyPatient with observed PSA + schedule.
        n_chains: number of independent chains for R-hat diagnostic. Default 3.
        n_steps: total steps per chain (including burn-in).
        burn_in: burn-in steps per chain.
        thin: thinning interval after burn-in.
        psa_noise_rel: relative noise model for likelihood (default 10%).
        prior_log_std: (6,) log-space prior std per parameter. Default 1.0 for
            r-rates and K_TP_drop, 0.5 for alpha entries (slightly tighter).
        theta_init_fn: callable rng -> theta_init for each chain. Default
            log-normal jitter around Zhang canonical.
        seed: top-level seed.
        verbose: log progress per chain.

    Returns:
        MCMCResult with all chains' samples and convergence diagnostics.
    """
    if prior_log_std is None:
        prior_log_std = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 1.0])
    prior_mean = _zhang_canonical_theta()
    sigma = psa_noise_rel * np.maximum(patient.psa_obs, 0.1 * np.max(patient.psa_obs))

    def log_post(theta: np.ndarray) -> float:
        return _log_posterior(theta, patient, sigma, prior_mean, prior_log_std)

    # Per-chain initial conditions: log-normal jitter around prior mean.
    rng_master = np.random.default_rng(seed)
    chain_seeds = rng_master.integers(0, 2**31 - 1, size=n_chains)
    if theta_init_fn is None:
        def theta_init_fn(rng_local):  # noqa: E731 — local def is fine
            return prior_mean * np.exp(rng_local.normal(0, 0.10, size=6))
    initial_proposal_scale = np.array([1e-4, 1e-4, 1e-4, 0.05, 0.05, 50.0])

    n_samples_per_chain = (n_steps - burn_in + thin - 1) // thin
    samples_all = np.zeros((n_chains, n_samples_per_chain, 6))
    accept_rates_all = np.zeros((n_chains, 6))
    log_post_traces = np.zeros((n_chains, n_steps))
    proposal_scales_final = np.zeros((n_chains, 6))
    theta_inits = np.zeros((n_chains, 6))

    for i, sd in enumerate(chain_seeds):
        rng_chain = np.random.default_rng(int(sd))
        theta_init = theta_init_fn(rng_chain)
        theta_inits[i] = theta_init
        if verbose:
            log.info(f"  chain {i}: starting at theta = {theta_init.tolist()}")
        samples, accept, trace, final_scale = _adaptive_mh_chain(
            log_post_fn=log_post,
            theta_init=theta_init,
            n_steps=n_steps,
            burn_in=burn_in,
            thin=thin,
            rng=rng_chain,
            proposal_scale_init=initial_proposal_scale.copy(),
        )
        # Pad/truncate to expected size
        n_actual = samples.shape[0]
        n_use = min(n_actual, n_samples_per_chain)
        samples_all[i, :n_use] = samples[:n_use]
        accept_rates_all[i] = accept
        log_post_traces[i] = trace
        proposal_scales_final[i] = final_scale
        if verbose:
            log.info(f"  chain {i}: accept rates = {accept.tolist()}")

    rhat = rhat_split(samples_all)
    return MCMCResult(
        samples=samples_all,
        param_names=PARAM_NAMES,
        rhat=rhat,
        accept_rates=accept_rates_all,
        log_posterior_traces=log_post_traces,
        proposal_scales_final=proposal_scales_final,
        theta_init=theta_inits,
        patient_id=patient.patient_id,
        n_total_steps=n_steps,
        burn_in=burn_in,
        thin=thin,
    )
