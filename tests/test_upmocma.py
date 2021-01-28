"""Testsuite for the :py:mod:`optimizers.upmocma` module."""
import unittest

import numpy as np

from anguilla.optimizers.upmocma import UPMOCMA


class TestUPMOCMA(unittest.TestCase):
    def test_initialization(self) -> None:
        """Test initialization of the UPMOCMA optimizer."""
        rng = np.random.default_rng()
        initial_points = rng.normal(size=(5, 4))
        initial_fitness = rng.normal(size=(5, 2))
        UPMOCMA(initial_points, initial_fitness, max_evaluations=1000)
