"""Testsuite for the utilitary module."""
import unittest

import numpy as np

from anguilla.util import exp_norm_chi, random_orthogonal_matrix


class TestUtilitaryFunctions(unittest.TestCase):
    """Test the utilitary functions."""

    def test_exp_norm_chi(self) -> None:
        """Test the exp_norm_chi function."""
        rng = np.random.default_rng(seed=0)

        def empirical(k):
            mean = np.zeros(k)
            cov = np.eye(k)
            Zs = rng.multivariate_normal(mean, cov, 1000000)
            Xs = np.linalg.norm(Zs, axis=1)
            return np.mean(Xs)

        emp_x = np.zeros(11 - 2 + 1)
        approx_x = np.zeros(11 - 2 + 1)

        for i, k in enumerate(range(2, 11)):
            emp_x[i] = empirical(k)
            approx_x[i] = exp_norm_chi(k)

        self.assertTrue(np.allclose(emp_x, approx_x, rtol=1e-2))


class TestRandomMatrices(unittest.TestCase):
    """Test generation of random matrices."""

    def setUp(self) -> None:
        """Set up for tests."""
        rng = np.random.default_rng(seed=0)
        self.Q = random_orthogonal_matrix(100, rng)

    def test_haar(self) -> None:
        """Check that the random matrix has a distribution that is a Haar \
        measure."""
        eig, _ = np.linalg.eig(self.Q)
        angles = np.angle(eig)
        density, _ = np.histogram(angles, bins="auto", density=True)
        self.assertTrue(np.allclose(density, 1.0 / (2.0 * np.pi), atol=1e-1))

    def test_orthogonal(self) -> None:
        """Check that the random matrix is orthogonal."""
        QQt = self.Q @ self.Q.T
        QtQ = self.Q.T @ self.Q
        I = np.eye(self.Q.shape[0])
        self.assertTrue(np.allclose(QQt, I))
        self.assertTrue(np.allclose(QtQ, I))
