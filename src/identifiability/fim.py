"""Fisher Information Matrix — finite-difference sensitivities.

Computes the FIM for a parameterized observation model:

    FIM[i, j] = sum_k (1/σ_k²) (∂y_k/∂θ_i)(∂y_k/∂θ_j)

where y is the predicted observation trajectory at sample times {t_k} and
σ_k is the observation noise std at time t_k. Sensitivities are computed
by central finite differences on a parameter vector θ.

Generic over the simulator: pass in a callable
``predict(theta) -> y_array`` and an array of nominal theta. The function
finite-differences each parameter (relative step ε by default) and assembles
the FIM matrix.

Rank deficiency of the FIM means that multiple parameter directions are
unidentifiable from the observation. The unidentifiable directions are the
right-singular vectors of the sensitivity matrix corresponding to small
singular values.

References:
- ``docs/notes/derivations.md`` Derivation 2 §"identifiability conjecture".
- ``docs/methodology/research_questions.md`` candidate C (identifiability +
  policy variation).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FIMResult:
    """Result of FIM computation.

    Attributes:
        fim: (P, P) Fisher information matrix.
        sensitivities: (P, T) sensitivity matrix S where S[i, k] = ∂y_k/∂θ_i.
        theta_nominal: (P,) parameter vector at which sensitivities were
            evaluated.
        param_names: optional list of parameter names (length P) for plot
            labels.
        sigma: noise std used for normalization (scalar or shape (T,)).
    """

    fim: np.ndarray
    sensitivities: np.ndarray
    theta_nominal: np.ndarray
    param_names: list[str] | None
    sigma: float | np.ndarray


def compute_fim(
    predict: Callable[[np.ndarray], np.ndarray],
    theta_nominal: np.ndarray,
    eps_rel: float = 1e-3,
    sigma: float | np.ndarray = 1.0,
    param_names: list[str] | None = None,
) -> FIMResult:
    """Compute the FIM via central finite-difference sensitivities.

    Args:
        predict: Callable that maps a parameter vector to an observation
            trajectory of shape (T,). Must be deterministic and
            differentiable in its arguments (numerically). For example,
            ``lambda theta: simulate_PSA_under_MTD(theta)``.
        theta_nominal: (P,) nominal parameter vector around which to
            compute sensitivities.
        eps_rel: Relative finite-difference step. For each parameter θ_i
            we use Δθ_i = max(eps_rel * |θ_i|, eps_rel) to handle zero or
            very small parameters. Central differences:
            ∂y/∂θ_i ≈ (y(θ + Δe_i) - y(θ - Δe_i)) / (2 Δ).
        sigma: Observation noise standard deviation. Scalar (assumed
            constant) or array of shape (T,). Default 1.0 = unit noise.
        param_names: Optional list of human-readable parameter names for
            plot labels.

    Returns:
        FIMResult with the (P, P) matrix and the (P, T) sensitivity matrix.

    Raises:
        ValueError: If theta_nominal has wrong shape, eps_rel is non-positive,
            or sigma has wrong shape.
    """
    theta = np.asarray(theta_nominal, dtype=float)
    if theta.ndim != 1:
        raise ValueError(f"theta_nominal must be 1-D; got shape {theta.shape}")
    if eps_rel <= 0:
        raise ValueError(f"eps_rel must be positive; got {eps_rel}")
    p = theta.size
    if param_names is not None and len(param_names) != p:
        raise ValueError(
            f"param_names length {len(param_names)} != theta size {p}"
        )

    # Predict at nominal to discover trajectory length.
    y_nom = np.asarray(predict(theta), dtype=float)
    if y_nom.ndim != 1:
        raise ValueError(f"predict must return 1-D array; got shape {y_nom.shape}")
    n_t = y_nom.size

    # Validate sigma shape.
    sigma_arr = np.asarray(sigma, dtype=float)
    if sigma_arr.ndim == 0:
        # scalar: broadcast
        pass
    elif sigma_arr.shape != (n_t,):
        raise ValueError(
            f"sigma must be scalar or shape ({n_t},); got {sigma_arr.shape}"
        )
    if np.any(sigma_arr <= 0):
        raise ValueError("sigma must be strictly positive everywhere")

    # Central differences for each parameter.
    S = np.zeros((p, n_t), dtype=float)
    for i in range(p):
        delta = max(eps_rel * abs(theta[i]), eps_rel)
        theta_plus = theta.copy()
        theta_plus[i] += delta
        theta_minus = theta.copy()
        theta_minus[i] -= delta

        y_plus = np.asarray(predict(theta_plus), dtype=float)
        y_minus = np.asarray(predict(theta_minus), dtype=float)
        if y_plus.shape != (n_t,) or y_minus.shape != (n_t,):
            raise RuntimeError(
                f"predict returned inconsistent shapes for parameter {i}"
            )
        S[i] = (y_plus - y_minus) / (2.0 * delta)

    # Normalize by sigma (broadcasts scalar or array).
    S_normalized = S / sigma_arr  # shape (P, T)
    fim = S_normalized @ S_normalized.T  # shape (P, P)

    # Symmetrize to mitigate floating-point noise.
    fim = 0.5 * (fim + fim.T)

    return FIMResult(
        fim=fim,
        sensitivities=S,
        theta_nominal=theta,
        param_names=param_names,
        sigma=sigma_arr if sigma_arr.ndim else float(sigma_arr),
    )


def fim_eigendecomposition(
    result: FIMResult,
    rank_threshold_rel: float = 1e-6,
) -> dict:
    """Eigendecompose the FIM and report identifiability structure.

    The FIM is symmetric positive-semidefinite. Eigenvalues are
    non-negative; small eigenvalues correspond to unidentifiable parameter
    directions (their corresponding eigenvectors).

    Args:
        result: FIMResult from compute_fim.
        rank_threshold_rel: Relative cutoff for "small" eigenvalues.
            Eigenvalues below rank_threshold_rel * max_eigenvalue are
            counted as unidentifiable.

    Returns:
        Dict with:
        - ``eigenvalues``: (P,) sorted descending.
        - ``eigenvectors``: (P, P) corresponding columns (sorted descending).
        - ``effective_rank``: int — count of eigenvalues above threshold.
        - ``most_identifiable_direction``: (P,) eigenvector for largest λ.
        - ``least_identifiable_direction``: (P,) eigenvector for smallest λ.
        - ``condition_number``: λ_max / max(λ_min, machine_eps).
    """
    fim = result.fim
    p = fim.shape[0]
    # eigh gives sorted eigenvalues ascending; we want descending for clarity.
    eigvals_asc, eigvecs_asc = np.linalg.eigh(fim)
    order = np.argsort(eigvals_asc)[::-1]
    eigvals = eigvals_asc[order]
    eigvecs = eigvecs_asc[:, order]
    # eigh can return tiny negative eigenvalues from float roundoff; clamp at 0.
    eigvals = np.clip(eigvals, 0.0, None)

    lam_max = float(eigvals[0])
    if lam_max == 0:
        eff_rank = 0
        cond = np.inf
    else:
        threshold = rank_threshold_rel * lam_max
        eff_rank = int(np.sum(eigvals > threshold))
        lam_min = max(eigvals[-1], np.finfo(float).eps)
        cond = lam_max / lam_min

    return {
        "eigenvalues": eigvals,
        "eigenvectors": eigvecs,
        "effective_rank": eff_rank,
        "n_params": p,
        "rank_deficient": eff_rank < p,
        "most_identifiable_direction": eigvecs[:, 0],
        "least_identifiable_direction": eigvecs[:, -1],
        "condition_number": cond,
    }
