"""Testsuite for the :py:mod:`optimizers.oplcma` module."""
import unittest

import numpy as np

from anguilla.optimizers.oplcma import OnePlusLambdaCMA
from anguilla.fitness.benchmark import Sphere


class TestOPLCMA(unittest.TestCase):
    """Test the OnePlusLambdaCMA optimizer."""

    def test_initialization(self) -> None:
        """Test initialization."""
        rng = np.random.default_rng()
        parent_point = rng.normal(size=5)
        parent_fitness = rng.normal(size=1)
        OnePlusLambdaCMA(parent_point, parent_fitness, 1, max_evaluations=1000)

    def test_fmin(self) -> None:
        """Test fmin (and implicitly ask and tell)."""
        rng = np.random.default_rng()
        fn = Sphere(3, rng=rng)
        parent_point = fn.random_points()
        parent_fitness = fn(parent_point)
        optimizer = OnePlusLambdaCMA(
            parent_point, parent_fitness, 1, max_evaluations=1500, rng=rng
        )
        result = optimizer.fmin(fn)
        self.assertTrue(result.solution.fitness < 1e-9)
