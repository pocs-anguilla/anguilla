"""Testsuite for the :py:mod:`optimizers` package."""
import math
import unittest

import numpy as np

from anguilla.optimizers import (
    MOParameters,
    MOStoppingConditions,
    SuccessNotion,
    cholesky_update,
)


class TestSuccessNotion(unittest.TestCase):
    def test_population_based(self) -> None:
        """Test that the population-based notion of success is defined."""
        sn = SuccessNotion.PopulationBased
        self.assertTrue(sn.value == SuccessNotion.PopulationBased.value)

    def test_individual_based(self) -> None:
        """Test that the individual-based notion of success is defined."""
        sn = SuccessNotion.IndividualBased
        self.assertTrue(sn.value == SuccessNotion.IndividualBased.value)


class TestMOParameters(unittest.TestCase):
    def test_default_values(self) -> None:
        """Test that the default values are those defined in the literature."""
        n = 3
        n_f = float(n)
        initial_step_size = 3.5
        parameters = MOParameters(n, initial_step_size)
        self.assertTrue(parameters.n_dimensions == n, "n_dimensions")
        self.assertTrue(parameters.n_offspring == 1, "n_offspring")
        self.assertTrue(
            parameters.initial_step_size == initial_step_size,
            "initial_step_size",
        )
        self.assertTrue(parameters.d == 1.0 + n_f / 2.0, "d")
        self.assertTrue(parameters.p_target_succ == 1.0 / 5.5, "p_target_succ")
        self.assertTrue(
            parameters.c_p
            == parameters.p_target_succ / (2.0 + parameters.p_target_succ),
            "c_p",
        )
        self.assertTrue(parameters.p_threshold == 0.44, "p_threshold")
        self.assertTrue(parameters.c_c == 2.0 / (n_f + 2.0), "c_c")
        self.assertTrue(
            math.isclose(parameters.c_cov, 2.0 / (n_f * n_f + 6.0)), "c_cov"
        )


class TestMOStoppingConditions(unittest.TestCase):
    def test_initialization(self):
        """Test initialization."""
        sc = MOStoppingConditions(max_evaluations=1000)
        self.assertTrue(sc.max_evaluations == 1000, "max_evaluations")
        self.assertFalse(sc.is_output, "is_output")
        self.assertFalse(sc.triggered, "triggered")

    def test_initialization_fails(self):
        """Test that initialization fails if no stopping conditions are \
        provided."""
        with self.assertRaises(ValueError):
            MOStoppingConditions()


class TestCholeskyUpdate(unittest.TestCase):
    def test_random(self):
        """Test using a randomly generated lower triangular matrix."""
        # We test that Lp is the Cholesky factor of alpha * X + beta * vv.T
        # Based on: https://git.io/JqaLJ and https://git.io/JqaCt

        rng = np.random.default_rng()
        n = 5
        A = rng.normal(size=(n, n))
        U, _, Vh = np.linalg.svd(np.dot(A.T, A))
        X = np.dot(np.dot(U, np.eye(n) * n), Vh)
        L = np.linalg.cholesky(X)

        v = rng.normal(size=(n,))
        alpha = rng.uniform(0.1, 1.0)
        beta = 1.0 - alpha

        Xp = alpha * X + beta * np.outer(v, v)
        expected_Lp = np.linalg.cholesky(Xp)
        Lp = cholesky_update(L, alpha, beta, v)

        self.assertTrue(
            Lp.shape == expected_Lp.shape,
            "shapes don't match, got: {}, expected: {}".format(
                Lp.shape, expected_Lp.shape
            ),
        )

        self.assertTrue(
            np.allclose(expected_Lp, Lp),
            "values don't match, got: {}, expected: {}".format(
                Lp, expected_Lp
            ),
        )
