"""Testsuite for the :py:mod:`optimizers` package."""
import pathlib
import math
import unittest
from typing import Any, Optional

import numpy as np
from numpy.linalg import cholesky

from anguilla.optimizers import (
    MOCMA,
    MOParameters,
    MOStoppingConditions,
    SuccessNotion,
    selection,
    cholesky_update,
)
from anguilla.dominance import non_dominated_sort
from anguilla.fitness import benchmark
from anguilla.fitness.base import ObjectiveFunction


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


class TestHypervolumeIndicatorSelection(unittest.TestCase):
    """Test hypervolume-indicator-based selection"""

    def test_example1(self):
        """Test a constructed 2D example"""
        # A set of non-dominated points
        points = np.array(
            [
                [0.01245897, 0.27127751],
                [0.02213313, 0.23395707],
                [0.0233907, 0.22994154],
                [0.0392689, 0.1886141],
                [0.04339422, 0.17990426],
                [0.16521067, 0.05107939],
                [0.17855283, 0.0440614],
                [0.28619405, 0.00950565],
            ]
        )
        expected = np.array(
            [True, False, False, False, False, True, False, True], dtype=bool
        )
        ranks, _ = non_dominated_sort(points)
        selected = selection(points, ranks, len(points) - 5)
        self.assertTrue(
            np.all(selected == expected),
            "got: {}, expected: {}".format(selected, expected),
        )

    def test_example2(self):
        """Test a constructed 2D example"""
        # A set of non-dominated points
        # A line with negative slope
        points = np.array(
            [
                [0.0, 0.0],
                [1.0, -1.0],
                [2.0, -2.0],
                [3.0, -3.0],
                [4.0, -4.0],
                [5.0, -5.0],
                [6.0, -6.0],
                [7.0, -7.0],
                [8.0, -8.0],
                [9.0, -9.0],
            ]
        )
        expected = np.array(
            [True, False, True, False, True, False, True, False, False, True],
            dtype=bool,
        )
        ranks, _ = non_dominated_sort(points)
        selected = selection(points, ranks, len(points) - 5)
        self.assertTrue(
            np.all(selected == expected),
            "got: {}, expected: {}".format(selected, expected),
        )

    def test_example3(self):
        """Test an example in which the first front has size one."""
        points = np.array(
            [
                [0.67996405, 0.29296719],
                [0.16359694, 0.67606755],
                [0.18515826, 0.91830604],
                [0.09758624, 0.21536309],
                [0.58368728, 0.52277089],
                [0.74548241, 0.51495986],
            ]
        )
        ranks, _ = non_dominated_sort(points)
        selected = selection(points, ranks, 3)
        expected = np.array([True, True, False, True, False, False])
        self.assertTrue(
            np.all(selected == expected),
            "got: {}, expected: {}".format(selected, expected),
        )


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
