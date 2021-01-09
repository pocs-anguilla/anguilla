"""Testsuite for the :py:mod:`util` module."""
import unittest

import numpy as np

from anguilla.util import random_orthogonal_matrix


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
        Qt = self.Q.T
        QQt = self.Q @ Qt
        QtQ = Qt @ self.Q
        Identity = np.eye(self.Q.shape[0])
        self.assertTrue(np.allclose(QQt, Identity))
        self.assertTrue(np.allclose(QtQ, Identity))
