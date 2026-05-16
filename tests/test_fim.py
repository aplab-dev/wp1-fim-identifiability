"""Tests for the FIM identifiability module.

Verifies:
- FIM is symmetric positive-semidefinite.
- Sensitivities match analytical derivatives for a known linear model.
- A redundantly parameterized model (e.g., y = (a+b)*t) produces a
  rank-deficient FIM with one zero eigenvalue (the (a-b) direction).
- Effective rank counting works at the threshold boundary.
- compute_fim raises on bad inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from identifiability import FIMResult, compute_fim, fim_eigendecomposition


# ---------- basic shape / sanity ----------


class TestFIMBasics:
    def test_returns_fim_result(self) -> None:
        def predict(theta: np.ndarray) -> np.ndarray:
            return theta[0] * np.linspace(0, 1, 10)

        r = compute_fim(predict, np.array([2.0]), param_names=["a"])
        assert isinstance(r, FIMResult)
        assert r.fim.shape == (1, 1)
        assert r.sensitivities.shape == (1, 10)

    def test_fim_is_symmetric(self) -> None:
        """FIM must be symmetric (within numerical precision)."""

        def predict(theta: np.ndarray) -> np.ndarray:
            t = np.linspace(0, 1, 20)
            return theta[0] * t + theta[1] * t**2 + theta[2] * np.sin(t)

        r = compute_fim(predict, np.array([1.0, 0.5, 0.3]))
        np.testing.assert_allclose(r.fim, r.fim.T, atol=1e-12)

    def test_fim_is_positive_semidefinite(self) -> None:
        """All eigenvalues of FIM are >= 0."""

        def predict(theta: np.ndarray) -> np.ndarray:
            t = np.linspace(0, 1, 30)
            return theta[0] * t + theta[1] * t**2 + theta[2] * np.exp(-t)

        r = compute_fim(predict, np.array([1.0, 0.5, 0.3]))
        eigvals = np.linalg.eigvalsh(r.fim)
        # Allow tiny negative values from float roundoff.
        assert np.all(eigvals >= -1e-10)


# ---------- analytical correctness ----------


class TestFIMAnalytical:
    def test_linear_model_sensitivity(self) -> None:
        """For y(t) = a*t, ∂y/∂a = t. FIM[0,0] = sum t² (with sigma=1)."""
        t = np.linspace(0, 1, 11)

        def predict(theta: np.ndarray) -> np.ndarray:
            return theta[0] * t

        r = compute_fim(predict, np.array([2.0]))
        # Sensitivity should equal t exactly.
        np.testing.assert_allclose(r.sensitivities[0], t, atol=1e-6)
        # FIM[0,0] = sum t².
        expected = np.sum(t**2)
        np.testing.assert_allclose(r.fim[0, 0], expected, rtol=1e-5)

    def test_sigma_scaling(self) -> None:
        """Doubling sigma quarters the FIM (since FIM ~ 1/sigma²)."""
        t = np.linspace(0, 1, 11)

        def predict(theta: np.ndarray) -> np.ndarray:
            return theta[0] * t

        r1 = compute_fim(predict, np.array([2.0]), sigma=1.0)
        r2 = compute_fim(predict, np.array([2.0]), sigma=2.0)
        np.testing.assert_allclose(r2.fim, r1.fim / 4.0, rtol=1e-5)


# ---------- rank deficiency ----------


class TestRankDeficiency:
    def test_redundant_parameterization(self) -> None:
        """For y(t) = (a+b)*t, the (a-b) direction is unidentifiable.

        FIM should have rank 1 in 2D parameter space — one nonzero eigenvalue
        plus one zero eigenvalue. Eigenvector for zero eigenvalue is along
        (1, -1)/sqrt(2).
        """
        t = np.linspace(0, 1, 50)

        def predict(theta: np.ndarray) -> np.ndarray:
            return (theta[0] + theta[1]) * t

        r = compute_fim(predict, np.array([1.0, 1.0]), param_names=["a", "b"])
        decomp = fim_eigendecomposition(r)

        assert decomp["effective_rank"] == 1
        assert decomp["rank_deficient"]
        # Least-identifiable direction should be (1, -1)/sqrt(2) up to sign.
        v = decomp["least_identifiable_direction"]
        # |v[0]| ≈ |v[1]| ≈ 1/sqrt(2)
        np.testing.assert_allclose(abs(v[0]), 1 / np.sqrt(2), atol=1e-6)
        np.testing.assert_allclose(abs(v[1]), 1 / np.sqrt(2), atol=1e-6)
        # opposite signs
        assert v[0] * v[1] < 0

    def test_full_rank_when_independent(self) -> None:
        """Two independently-acting parameters should give a rank-2 FIM."""
        t = np.linspace(0, 1, 50)

        def predict(theta: np.ndarray) -> np.ndarray:
            return theta[0] * t + theta[1] * t**2

        r = compute_fim(predict, np.array([1.0, 1.0]))
        decomp = fim_eigendecomposition(r)
        assert decomp["effective_rank"] == 2
        assert not decomp["rank_deficient"]


# ---------- input validation ----------


class TestFIMValidation:
    def test_bad_theta_shape_rejected(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            compute_fim(lambda th: th[0] * np.ones(5), np.array([[1.0, 2.0]]))

    def test_non_positive_eps_rejected(self) -> None:
        with pytest.raises(ValueError, match="eps_rel"):
            compute_fim(
                lambda th: th[0] * np.ones(5),
                np.array([1.0]),
                eps_rel=-0.01,
            )

    def test_sigma_wrong_shape_rejected(self) -> None:
        def predict(th: np.ndarray) -> np.ndarray:
            return th[0] * np.linspace(0, 1, 10)

        with pytest.raises(ValueError, match="sigma"):
            compute_fim(predict, np.array([1.0]), sigma=np.array([1.0, 2.0]))

    def test_negative_sigma_rejected(self) -> None:
        def predict(th: np.ndarray) -> np.ndarray:
            return th[0] * np.linspace(0, 1, 10)

        with pytest.raises(ValueError, match="positive"):
            compute_fim(predict, np.array([1.0]), sigma=-1.0)

    def test_param_names_wrong_length_rejected(self) -> None:
        with pytest.raises(ValueError, match="param_names"):
            compute_fim(
                lambda th: th[0] * np.ones(5),
                np.array([1.0, 2.0]),
                param_names=["a"],
            )

    def test_predict_returns_2d_rejected(self) -> None:
        def predict(th: np.ndarray) -> np.ndarray:
            return np.zeros((3, 4))

        with pytest.raises(ValueError, match="1-D"):
            compute_fim(predict, np.array([1.0]))


# ---------- eigendecomposition ----------


class TestEigendecomposition:
    def test_eigenvalues_are_descending(self) -> None:
        t = np.linspace(0, 1, 20)

        def predict(theta: np.ndarray) -> np.ndarray:
            return theta[0] * t + theta[1] * t**2 + theta[2] * np.sin(t)

        r = compute_fim(predict, np.array([1.0, 0.5, 0.3]))
        decomp = fim_eigendecomposition(r)
        eigvals = decomp["eigenvalues"]
        diffs = np.diff(eigvals)
        assert np.all(diffs <= 1e-10), "eigenvalues must be descending"

    def test_condition_number_finite_for_full_rank(self) -> None:
        t = np.linspace(0, 1, 20)

        def predict(theta: np.ndarray) -> np.ndarray:
            return theta[0] * t + theta[1] * t**2

        r = compute_fim(predict, np.array([1.0, 1.0]))
        decomp = fim_eigendecomposition(r)
        assert np.isfinite(decomp["condition_number"])
        assert decomp["condition_number"] > 0
