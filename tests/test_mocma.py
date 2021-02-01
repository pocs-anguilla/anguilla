"""Testsuite for the :py:mod:`optimizers.mocma` module."""
import pathlib
import unittest
from typing import Any, Optional

import numpy as np

from anguilla.fitness import benchmark
from anguilla.fitness.base import ObjectiveFunction
from anguilla.optimizers.mocma import MOCMA

VOLUME_TEST_N_TRIALS = 3
VOLUME_TEST_RTOL = 5e-3


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
                # Should check this
                reference_volume = np.max(volumes)
                self.assertTrue(
                    np.allclose(
                        reference_volume,
                        target_volume,
                        rtol=VOLUME_TEST_RTOL,
                    ),
                    f"Failed (row {i}), got {reference_volume}, expected {target_volume}",
                )

    def test_volume_individual(self) -> None:
        self.run_test_volume("individual")

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
