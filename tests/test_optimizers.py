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
from anguilla.indicators import HypervolumeIndicator
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


class TestMOCMA(unittest.TestCase):
    def test_initialization(self):
        """Test initialization."""
        rng = np.random.default_rng(0)
        points = rng.uniform(size=(3, 5))
        fitness = rng.uniform(size=(3, 2))
        optimizer = MOCMA(
            points,
            fitness,
            success_notion="individual",
            max_evaluations=1000,
        )
        self.assertTrue(optimizer.name == "MO-CMA-ES", "name")
        self.assertTrue(
            optimizer.qualified_name == "(3+3)-MO-CMA-ES-I", "qualified_name"
        )
        self.assertTrue(
            optimizer.success_notion.value
            == SuccessNotion.IndividualBased.value,
            "success_notion",
        )
        self.assertTrue(optimizer.generation_count == 0, "generation_count")
        self.assertTrue(optimizer.evaluation_count == 0, "evaluation_count")
        self.assertTrue(
            optimizer.parameters.n_dimensions == 5, "parameters.n_dimensions"
        )
        self.assertTrue(
            optimizer.stopping_conditions.max_generations == None,
            "stopping_conditions.max_generations",
        )
        self.assertTrue(
            optimizer.stopping_conditions.max_evaluations == 1000,
            "stopping_conditions.max_evaluations",
        )
        self.assertTrue(
            np.all(optimizer.population.point[:3] == points), "points"
        )
        self.assertTrue(
            np.all(optimizer.population.fitness[:3] == fitness), "fitness"
        )

    def test_initialization_steady(self):
        """Test initialization for the steady-state variant."""
        rng = np.random.default_rng(0)
        points = rng.uniform(size=(3, 5))
        fitness = rng.uniform(size=(3, 2))
        optimizer = MOCMA(
            points,
            fitness,
            max_evaluations=1000,
            n_offspring=1,
        )
        self.assertTrue(optimizer.name == "MO-CMA-ES", "name")
        self.assertTrue(
            optimizer.qualified_name == "(3+1)-MO-CMA-ES-P", "qualified_name"
        )
        self.assertTrue(
            optimizer.success_notion.value
            == SuccessNotion.PopulationBased.value,
            "success_notion",
        )
        self.assertTrue(optimizer.generation_count == 0, "generation_count")
        self.assertTrue(optimizer.evaluation_count == 0, "evaluation_count")
        self.assertTrue(
            optimizer.parameters.n_dimensions == 5, "parameters.n_dimensions"
        )
        self.assertTrue(
            optimizer.stopping_conditions.max_generations == None,
            "stopping_conditions.max_generations",
        )
        self.assertTrue(
            optimizer.stopping_conditions.max_evaluations == 1000,
            "stopping_conditions.max_evaluations",
        )
        self.assertTrue(
            np.all(optimizer.population.point[:3] == points), "points"
        )
        self.assertTrue(
            np.all(optimizer.population.fitness[:3] == fitness), "fitness"
        )

    def test_best(self):
        """Test that the best method works as expected."""
        rng = np.random.default_rng(0)
        points = rng.uniform(size=(3, 5))
        fitness = rng.uniform(size=(3, 2))
        optimizer = MOCMA(
            points,
            fitness,
            success_notion="individual",
            max_evaluations=1000,
        )
        best = optimizer.best
        self.assertTrue(np.all(best.point == points), "points")
        self.assertTrue(np.all(best.fitness == fitness), "fitness")
        self.assertTrue(
            np.all(
                best.step_size
                == np.repeat(optimizer.parameters.initial_step_size, 3)
            ),
            "step_size",
        )

    def test_ask(self):
        """Test that the ask method runs without errors."""
        rng = np.random.default_rng(0)
        n = 100
        points = rng.uniform(size=(n, 5))
        fitness = rng.uniform(size=(n, 2))
        optimizer = MOCMA(
            points,
            fitness,
            success_notion="individual",
            max_evaluations=1000,
        )
        points = optimizer.ask()
        # Basic test shape
        self.assertTrue(points.shape == (n, 5), "points shape")
        # Test that the parent indices are correct
        result = optimizer.population.parent_index
        expected = np.arange(0, n, dtype=int)
        self.assertTrue(
            np.all(result == expected),
            "parent indices, got: {}, expected: {}".format(result, expected),
        )
        # Test that no numbers are infinite or NaN
        self.assertFalse(
            np.any(np.isinf(points)),
            "Got infinite values: {}".format(points),
        )
        self.assertFalse(
            np.any(np.isnan(points)),
            "Got NaN values: {}".format(points),
        )
        # Test that the mutation works as expected
        result = points[0]
        expected = optimizer.population.point[
            0
        ] + optimizer.population.step_size[0] * (
            optimizer.population.cov[0] @ optimizer.population.z[0]
        )
        self.assertTrue(
            np.allclose(
                result,
                expected,
            ),
            "mutation: got {}, expected: {}".format(result, expected),
        )

    def test_ask_variant(self):
        """Test the ask method for the n_offspring != n_parents variant."""
        rng = np.random.default_rng(0)
        n_parents = 8
        n_offspring = 4
        points = rng.uniform(size=(n_parents, 5))
        # In this set all points have different rank
        # Only the first element has rank 1
        fitness = np.array(
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
        optimizer = MOCMA(
            points,
            fitness,
            max_evaluations=1000,
            n_offspring=n_offspring,
        )
        # Test that the parent indices are correct
        result = optimizer.population.parent_index
        expected = np.zeros(n_offspring, dtype=int)
        self.assertTrue(
            np.all(result == expected),
            "parent indices, got: {}, expected: {}".format(result, expected),
        )

    def test_tell(self):
        """Test that the tell method runs without errors."""
        # Note: this doesn't test anything about the correctness
        # of adaptation and selection, just that they don't result in an error.
        n = 3
        rng = np.random.default_rng()
        points = rng.uniform(size=(n, 5))
        fitness = rng.uniform(size=(n, 2))
        optimizer = MOCMA(
            points,
            fitness,
            success_notion="individual",
            max_evaluations=1000,
        )
        points = optimizer.ask()
        fitness = rng.uniform(size=(n, 2))
        penalized_fitness = rng.uniform(size=(n, 2))
        optimizer.tell(fitness, penalized_fitness)
        # We test the offspring data is copied correctly
        # which we can do since the offspring data remains untouched
        # in its buffer after selection.
        self.assertTrue(
            np.all(fitness == optimizer.population.fitness[n:]), "fitness"
        )
        self.assertTrue(
            np.all(
                penalized_fitness == optimizer.population.penalized_fitness[n:]
            ),
            "penalized_fitness",
        )
        self.assertTrue(optimizer.generation_count == 1, "generation_count")
        self.assertTrue(optimizer.evaluation_count == n, "evaluation_count")


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


class VolumeBaseTestFunction:
    """Test the MOCMA implementation against known volumes."""

    filename: str
    fn_cls: ObjectiveFunction
    n_offspring: Optional[int] = None
    min_n_parents: Optional[int] = None
    max_n_parents: Optional[int] = None

    assertTrue: Any

    def setUp(self):
        path = (
            pathlib.Path(__file__)
            .parent.joinpath("data/volumes")
            .joinpath(self.filename)
            .absolute()
        )
        self.data = np.genfromtxt(str(path), delimiter=",")
        self.rng = np.random.default_rng()

    def get_fn(self) -> ObjectiveFunction:
        raise NotImplementedError()

    def run_test_volume(self, success_notion: str) -> None:
        for i, row in enumerate(self.data):
            n_parents = int(row[0])
            if (
                not self.max_n_parents or n_parents <= self.max_n_parents
            ) and (not self.min_n_parents or n_parents >= self.min_n_parents):
                target_volume = row[1]
                max_evaluations = int(row[2])
                n_dimensions = int(row[3])
                n_objectives = int(row[4])
                reference = row[5 : 5 + n_objectives]
                volumes = np.empty(1)
                fn = self.fn_cls(rng=self.rng)
                fn.n_dimensions = n_dimensions
                fn.n_objectives = n_objectives

                indicator = HypervolumeIndicator(reference)
                for trial in range(1):
                    parent_points = fn.random_points(n_parents)
                    parent_fitness = fn(parent_points)
                    optimizer = MOCMA(
                        parent_points,
                        parent_fitness,
                        n_offspring=self.n_offspring,
                        success_notion=success_notion,
                        max_evaluations=max_evaluations,
                        seed=self.rng.integers(0, 10000),
                    )
                    while not optimizer.stop.triggered:
                        points = optimizer.ask()
                        if fn.has_constraints:
                            optimizer.tell(*fn.evaluate_with_penalty(points))
                        else:
                            optimizer.tell(fn(points))
                    volumes[trial] = indicator(optimizer.best.fitness)
                reference_volume = np.median(volumes)
                self.assertTrue(
                    np.allclose(
                        reference_volume,
                        target_volume,
                        rtol=5e-3,
                    ),
                    "Failed (row {}), got {}, expected {}".format(
                        i, reference_volume, target_volume
                    ),
                )

    # TODO: Check if individual notion of success needs more evaluations
    #       for this unit test
    # def test_volume_individual(self) -> None:
    #    self.run_test_volume("individual")

    def test_volume_population(self) -> None:
        self.run_test_volume("population")


class TestMOCMAVolumeZDT1(VolumeBaseTestFunction, unittest.TestCase):
    """Unit tests for the MOCMA optimizer with the ZDT1 function."""

    filename = "ZDT1.csv"
    fn_cls = benchmark.ZDT1
    min_n_parents = 10
    max_n_parents = 10
