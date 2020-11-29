"""Testsuite for the CMA module."""
import math
import unittest

import numpy as np

from anguilla.optimizers.cma import StrategyParameters, CMA


# TODO: WIP
class TestCMAStrategyParameters(unittest.TestCase):
    """Test strategy parameters."""

    def setUp(self) -> None:
        """Set up for tests."""
        self.params = StrategyParameters(n=8, active=True)

    def test_positive_weights(self) -> None:
        """Test post-condition for positive weights."""
        mu = self.params.mu
        weights = self.params.weights

        self.assertTrue(math.isclose(1.0, np.sum(weights[:mu])))

    def test_negative_weights(self) -> None:
        """Test post-condition for negative weights."""
        # TODO: Failing except for default weights when using n in {7,8}
        #       [2016:cma-tutorial-es] says negative weights _usually_ sum
        #       to -alpha_mu_eff_neg so maybe remove this unit test?
        mu = self.params.mu
        mu_eff = self.params.mu_eff
        mu_eff_neg = self.params.mu_eff_neg
        weights = self.params.weights

        alpha_mu_eff_neg = 1.0 + (2.0 * mu_eff) / (mu_eff_neg + 2.0)
        expected_value = -alpha_mu_eff_neg
        true_value = np.sum(weights[mu:])

        self.assertTrue(
            math.isclose(expected_value, true_value),
            msg=f"Expected {expected_value}, got {true_value}",
        )


# TODO: WIP
class TestCMAOptimizer(unittest.TestCase):
    """Test the CMA optimizer."""

    def test_init_1(self) -> None:
        """Test initialization."""
        initial_point = np.random.uniform(size=10)
        _ = CMA(initial_point=initial_point, initial_sigma=0.5)

    def test_init_2(self) -> None:
        """Test initialization."""
        params = StrategyParameters(n=10)
        initial_point = np.random.uniform(size=10)
        C = np.eye(10)
        _ = CMA(
            initial_point=initial_point,
            initial_sigma=0.5,
            initial_cov=C,
            strategy_parameters=params,
        )

    def test_ask(self) -> None:
        """Test the ask method."""
        # TODO: Fix this test
        n = 3
        sigma = 0.5
        rng = np.random.default_rng(seed=0)
        initial_point = rng.uniform(size=n)

        rng = np.random.default_rng(seed=0)
        C = np.eye(n)
        opt = CMA(
            initial_point=initial_point,
            initial_sigma=sigma,
            initial_cov=C,
            rng=rng,
        )

        solutions_0 = opt.ask()

        rng = np.random.default_rng(seed=0)
        solutions_1 = rng.multivariate_normal(
            initial_point,
            (sigma ** 2) * C,
            size=opt._params.population_size,
            method="eigh",
            check_valid="raise",
        )

        self.assertTrue(np.allclose(solutions_1, solutions_0))
