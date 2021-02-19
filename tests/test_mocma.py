"""Testsuite for the :py:mod:`optimizers.mocma` module."""
import pathlib
import unittest
from typing import Any, Optional

import numpy as np

from anguilla.fitness import benchmark
from anguilla.fitness.base import ObjectiveFunction
from anguilla.optimizers.mocma import MOCMA, FixedSizePopulation

VOLUME_TEST_N_TRIALS = 1
VOLUME_TEST_RTOL = 5e-3


class TestFixedSizePopulation(unittest.TestCase):
    """Test the fixed-sized population container."""

    def test_initialization(self):
        """Test initialization."""
        rng = np.random.default_rng()
        n_parents = 10
        n_offspring = 5
        n_total = n_parents + n_offspring
        n_dimensions = 5
        n_objectives = 3
        population = FixedSizePopulation(
            n_dimensions,
            n_objectives,
            n_parents,
            n_offspring,
        )
        self.assertTrue(
            population.fitness.shape == (n_total, n_objectives),
            "fitness shape",
        )
        self.assertTrue(
            population.points.shape == (n_total, n_dimensions), "points shape"
        )
        self.assertTrue(
            population.cov.shape == (n_total, n_dimensions, n_dimensions),
            "Cov. matrices shape",
        )
        index = rng.integers(0, n_total)
        self.assertTrue(
            np.all(population.cov[index] == np.eye(n_dimensions)),
            "Cov. matrices initialization, got: {}".format(
                population.cov[index],
            ),
        )

    def test_update_views(self):
        """Test that updating data using views works as expected."""
        rng = np.random.default_rng()
        n_objectives = 3
        population = FixedSizePopulation(5, n_objectives, 10, 10)
        fitness_view = population.fitness[:]
        new_fitness_chunk = rng.uniform(0, 35, size=(4, n_objectives))
        fitness_view[3:7] = new_fitness_chunk
        self.assertTrue(
            np.all(population.fitness[3:7] == new_fitness_chunk),
            "first update to fitness_view",
        )
        fitness_view[3:7] = rng.uniform(36, 105, size=(4, n_objectives))
        self.assertFalse(
            np.all(population.fitness[3:7] == new_fitness_chunk),
            "second update to fitness_view",
        )
        index = rng.integers(0, 20)
        p_succ_view = population.p_succ[:]
        p_succ_view[index] = 0.5
        self.assertTrue(
            population.p_succ[index] == 0.5, "first update to p_succ_view"
        )
        population.p_succ[index] = 1.8
        self.assertTrue(p_succ_view[index] == 1.8, "second update to p_succ")

    def test_update_cov(self):
        """Test that updating cov. matrices using views works as expected."""
        rng = np.random.default_rng()
        n_dimensions = 5
        n_parents = 10
        n_offspring = 10
        population = FixedSizePopulation(
            n_dimensions, 3, n_parents, n_offspring
        )
        oidx = rng.integers(0, n_parents)
        pidx = rng.integers(n_parents, n_parents + n_offspring)
        population.cov[pidx, :, :] = rng.uniform(
            0,
            100,
            size=(n_dimensions, n_dimensions),
        )
        population.cov[oidx, :, :] = population.cov[pidx, :, :]
        self.assertTrue(
            np.all(population.cov[oidx] == population.cov[pidx]),
            "copy from parent to offspring",
        )
        population.cov[oidx, :, :] = rng.uniform(
            100,
            200,
            size=(n_dimensions, n_dimensions),
        )
        self.assertFalse(
            np.all(population.cov[oidx] == population.cov[pidx]),
            "update offspring",
        )


class BasicTests(unittest.TestCase):
    """Test basic properties."""

    def test_evaluation_count_steady(self):
        fn = benchmark.GELLI(5, 3)
        parent_points = fn.random_points(10)
        parent_fitness = fn(parent_points)
        optimizer = MOCMA(
            parent_points, parent_fitness, n_offspring=1, max_evaluations=4
        )
        res = optimizer.fmin(fn)
        self.assertTrue(
            res.stopping_conditions.max_evaluations == 4,
            "stopping condition count, got: {}, expected: {}".format(
                res.stopping_conditions.max_evaluations, 4
            ),
        )
        self.assertTrue(
            fn.evaluation_count - len(parent_points)
            == optimizer.evaluation_count,
            "function count: {}, optimizer count: {}".format(
                fn.evaluation_count - len(parent_points),
                optimizer.evaluation_count,
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
                volumes = np.empty(VOLUME_TEST_N_TRIALS)
                fn = self.fn_cls(rng=self.rng)
                fn.n_dimensions = n_dimensions
                fn.n_objectives = n_objectives

                # We will call fmin with evaluate
                def evaluate(points):
                    if fn.has_constraints:
                        return fn.evaluate_with_penalty(points)
                    return fn(points)

                for trial in range(VOLUME_TEST_N_TRIALS):
                    parent_points = fn.random_points(n_parents)
                    parent_fitness = fn(parent_points)
                    optimizer = MOCMA(
                        parent_points,
                        parent_fitness,
                        n_offspring=self.n_offspring,
                        success_notion=success_notion,
                        max_evaluations=max_evaluations,
                        rng=self.rng,
                    )
                    optimizer.indicator.reference = reference

                    result = optimizer.fmin(evaluate)

                    volumes[trial] = optimizer.indicator(
                        result.solution.fitness
                    )
                reference_volume = np.median(volumes)
                self.assertTrue(
                    np.allclose(
                        reference_volume,
                        target_volume,
                        rtol=VOLUME_TEST_RTOL,
                    ),
                    f"Failed (row {i}), got {reference_volume}, expected {target_volume}",
                )

    # TODO: Check if individual notion of success needs more evaluations
    #       for this unit test
    #def test_volume_individual(self) -> None:
    #    self.run_test_volume("individual")

    def test_volume_population(self) -> None:
        self.run_test_volume("population")


class TestMOCMAVolumeZDT1(VolumeBaseTestFunction, unittest.TestCase):
    """Unit tests for the MOCMA optimizer with the ZDT1 function."""

    filename = "ZDT1.csv"
    fn_cls = benchmark.ZDT1
    min_n_parents = 10
    max_n_parents = 10


class TestMOCMAVolumeZDT2(VolumeBaseTestFunction, unittest.TestCase):
    """Unit tests for the MOCMA optimizer with the ZDT2 function."""

    filename = "ZDT2.csv"
    fn_cls = benchmark.ZDT2
    min_n_parents = 10
    max_n_parents = 10


class TestMOCMAVolumeZDT3(VolumeBaseTestFunction, unittest.TestCase):
    """Unit tests for the MOCMA optimizer with the ZDT3 function."""

    filename = "ZDT3.csv"
    fn_cls = benchmark.ZDT3
    min_n_parents = 10
    max_n_parents = 10


class TestMOCMAVolumeZDT6(VolumeBaseTestFunction, unittest.TestCase):
    """Unit tests for the MOCMA optimizer with the ZDT6 function."""

    filename = "ZDT6.csv"
    fn_cls = benchmark.ZDT6
    min_n_parents = 10
    max_n_parents = 10


class TestMOCMAVolumeDTLZ2(VolumeBaseTestFunction, unittest.TestCase):
    """Unit tests for the MOCMA optimizer with the DTLZ2 function."""

    filename = "DTLZ2.csv"
    fn_cls = benchmark.DTLZ2
    min_n_parents = 10
    max_n_parents = 10


class TestMOCMAVolumeDTLZ4(VolumeBaseTestFunction, unittest.TestCase):
    """Unit tests for the MOCMA optimizer with the DTLZ4 function."""

    filename = "DTLZ4.csv"
    fn_cls = benchmark.DTLZ4
    min_n_parents = 10
    max_n_parents = 10
