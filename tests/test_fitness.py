"""Testsuite for the :py:mod:`fitness.benchmark` package."""
import dataclasses
import pathlib
import unittest
from typing import Any

import numpy as np

from anguilla.fitness.base import ObjectiveFunction
import anguilla.fitness.benchmark as benchmark


@dataclasses.dataclass
class Samples:
    """A set of sample data."""

    points: np.ndarray
    fitness: np.ndarray
    n_dimensions: int
    n_objectives: int


def get_sample(row: np.ndarray) -> Samples:
    start_point, end_point, start_fitness, end_fitness = row[0:4]
    point = np.squeeze(row[int(start_point) : int(end_point)])
    fitness = np.squeeze(row[int(start_fitness) : int(end_fitness)])
    n_dimensions = int(end_point - start_point)
    n_objectives = int(end_fitness - start_fitness)
    return Samples(point, fitness, n_dimensions, n_objectives)


def get_samples(rows: np.ndarray) -> Samples:
    # Assumes all samples in the given rows have the same number of
    # dimensions and objectives
    start_point, end_point, start_fitness, end_fitness = rows[0, 0:4]
    points = np.squeeze(rows[:, int(start_point) : int(end_point)])
    fitness = np.squeeze(rows[:, int(start_fitness) : int(end_fitness)])
    n_dimensions = int(end_point - start_point)
    n_objectives = int(end_fitness - start_fitness)
    return Samples(points, fitness, n_dimensions, n_objectives)


class BaseTestFunction:
    """Base test case for benchmark functions.

    Notes
    -----
    Subclasses use the inheritance approach suggested \
    `here <https://stackoverflow.com/a/52261358>`_.
    """

    fn: ObjectiveFunction
    filename: str

    assertTrue: Any

    def setUp(self):
        path = (
            pathlib.Path(__file__)
            .parent.joinpath("data/fitness")
            .joinpath(self.filename)
            .absolute()
        )
        self.data = np.genfromtxt(str(path), delimiter=",")

    def get_fn(self):
        raise NotImplementedError()

    def test_scale_dimensions(self):
        fn = self.get_fn()
        start = fn.n_dimensions
        fn.n_dimensions = start + 1
        end = fn.n_dimensions
        if fn.scalable_dimensions:
            self.assertTrue(start != end, "!= operator")
        else:
            self.assertTrue(start == end, "== operator")

    def test_scale_objectives(self):
        fn = self.get_fn()
        start = fn.n_objectives
        fn.n_objectives = start + 1
        end = fn.n_objectives
        if fn.scalable_objectives:
            self.assertTrue(start != end, "!= operator")
        else:
            self.assertTrue(start == end, "== operator")

    def test_single(self):
        fn = self.get_fn()
        for i, row in enumerate(self.data):
            sample = get_sample(row)
            fn.n_dimensions = sample.n_dimensions
            fn.n_objectives = sample.n_objectives
            self.assertTrue(
                np.allclose(fn.evaluate_single(sample.points), sample.fitness),
                f"Values don't match (row {i})",
            )

    def test_multiple(self):
        fn = self.get_fn()
        samples = get_samples(self.data[5:])
        fn.n_dimensions = samples.n_dimensions
        fn.n_objectives = samples.n_objectives
        self.assertTrue(
            np.allclose(fn.evaluate_multiple(samples.points), samples.fitness)
        )

    def test_single_multiple(self):
        fn = self.get_fn()
        for row in self.data:
            sample = get_sample(row)
            fn.n_dimensions = sample.n_dimensions
            fn.n_objectives = sample.n_objectives
            v1 = fn.evaluate_single(sample.points)
            v2 = fn.evaluate_multiple(sample.points)
            self.assertTrue(v1.shape == v2.shape, "Shapes don't match.")
            self.assertTrue(np.allclose(v1, v2), "Values don't match.")

    def test_call(self):
        fn = self.get_fn()
        samples = get_samples(self.data[5:])
        fn.n_dimensions = samples.n_dimensions
        fn.n_objectives = samples.n_objectives
        self.assertTrue(
            np.allclose(fn(samples.points), samples.fitness),
            "Call with multiple failed.",
        )
        for point, fitness in zip(samples.points, samples.fitness):
            self.assertTrue(
                np.allclose(fn(point), fitness), "Call with single failed."
            )

    def test_evaluation_count(self):
        fn = self.get_fn()
        fn.n_dimensions = 5
        self.assertTrue(
            fn.evaluation_count == 0, "Initial evaluation count failed."
        )
        fn(fn.random_points(1))
        self.assertTrue(
            fn.evaluation_count == 1,
            f"Evaluation count after single call failed, got {fn.evaluation_count}.",
        )
        fn(fn.random_points(11))
        self.assertTrue(
            fn.evaluation_count == 12,
            f"Evaluation count after multiple call failed, got {fn.evaluation_count}.",
        )


class TestSphere(BaseTestFunction, unittest.TestCase):
    """Unit tests for the Sphere function."""

    filename = "Sphere.csv"

    def get_fn(self) -> benchmark.Sphere:
        return benchmark.Sphere()


class TestRastrigin(BaseTestFunction, unittest.TestCase):
    """Unit tests for the Rastrigin function."""

    filename = "Rastrigin.csv"

    def get_fn(self) -> benchmark.Rastrigin:
        return benchmark.Rastrigin(rotate=False)


class TestEllipsoid(BaseTestFunction, unittest.TestCase):
    """Unit tests for the Ellipsoid function."""

    filename = "Ellipsoid.csv"

    def get_fn(self) -> benchmark.Ellipsoid:
        return benchmark.Ellipsoid(rotate=False)


class TestZDT1(BaseTestFunction, unittest.TestCase):
    """Unit tests for the ZDT1 function."""

    filename = "ZDT1.csv"

    def get_fn(self) -> benchmark.ZDT1:
        return benchmark.ZDT1()


class TestZDT2(BaseTestFunction, unittest.TestCase):
    """Unit tests for the ZDT2 function."""

    filename = "ZDT2.csv"

    def get_fn(self) -> benchmark.ZDT2:
        return benchmark.ZDT2()


class TestZDT3(BaseTestFunction, unittest.TestCase):
    """Unit tests for the ZDT3 function."""

    filename = "ZDT3.csv"

    def get_fn(self) -> benchmark.ZDT3:
        return benchmark.ZDT3()


class TestZDT4(BaseTestFunction, unittest.TestCase):
    """Unit tests for the ZDT4 function."""

    filename = "ZDT4.csv"

    def get_fn(self) -> benchmark.ZDT4:
        return benchmark.ZDT4()


class TestZDT6(BaseTestFunction, unittest.TestCase):
    """Unit tests for the ZDT6 function."""

    filename = "ZDT6.csv"

    def get_fn(self) -> benchmark.ZDT6:
        return benchmark.ZDT6()


class TestIHR1(BaseTestFunction, unittest.TestCase):
    """Unit tests for the IHR1 function."""

    filename = "IHR1.csv"

    def get_fn(self) -> benchmark.IHR1:
        return benchmark.IHR1(rotate=False)


class TestIHR2(BaseTestFunction, unittest.TestCase):
    """Unit tests for the IHR2 function."""

    filename = "IHR2.csv"

    def get_fn(self) -> benchmark.IHR2:
        return benchmark.IHR2(rotate=False)


class TestELLI1(BaseTestFunction, unittest.TestCase):
    """Unit tests for the ELLI1 function."""

    filename = "ELLI1.csv"

    def get_fn(self) -> benchmark.ELLI1:
        return benchmark.ELLI1(rotate=False)


class TestELLI2(BaseTestFunction, unittest.TestCase):
    """Unit tests for the ELLI2 function."""

    filename = "ELLI2.csv"

    def get_fn(self) -> benchmark.ELLI2:
        return benchmark.ELLI2(rotate=False)


class TestCIGTAB1(BaseTestFunction, unittest.TestCase):
    """Unit tests for the CIGTAB1 function."""

    filename = "CIGTAB1.csv"

    def get_fn(self) -> benchmark.CIGTAB1:
        return benchmark.CIGTAB1(rotate=False)


class TestCIGTAB2(BaseTestFunction, unittest.TestCase):
    """Unit tests for the CIGTAB2 function."""

    filename = "CIGTAB2.csv"

    def get_fn(self) -> benchmark.CIGTAB2:
        return benchmark.CIGTAB2(rotate=False)


class TestDTLZ1(BaseTestFunction, unittest.TestCase):
    """Unit tests for the DTLZ1 function."""

    filename = "DTLZ1.csv"

    def get_fn(self) -> benchmark.DTLZ1:
        return benchmark.DTLZ1(5)


class TestDTLZ2(BaseTestFunction, unittest.TestCase):
    """Unit tests for the DTLZ2 function."""

    filename = "DTLZ2.csv"

    def get_fn(self) -> benchmark.DTLZ2:
        return benchmark.DTLZ2(5)


class TestDTLZ3(BaseTestFunction, unittest.TestCase):
    """Unit tests for the DTLZ3 function."""

    filename = "DTLZ3.csv"

    def get_fn(self) -> benchmark.DTLZ3:
        return benchmark.DTLZ3(5)


class TestDTLZ4(BaseTestFunction, unittest.TestCase):
    """Unit tests for the DTLZ4 function."""

    filename = "DTLZ4.csv"

    def get_fn(self) -> benchmark.DTLZ4:
        return benchmark.DTLZ4(5)


class TestDTLZ5(BaseTestFunction, unittest.TestCase):
    """Unit tests for the DTLZ5 function."""

    filename = "DTLZ5.csv"

    def get_fn(self) -> benchmark.DTLZ5:
        return benchmark.DTLZ5(5)


class TestDTLZ6(BaseTestFunction, unittest.TestCase):
    """Unit tests for the DTLZ6 function."""

    filename = "DTLZ6.csv"

    def get_fn(self) -> benchmark.DTLZ6:
        return benchmark.DTLZ6(5)


class TestDTLZ7(BaseTestFunction, unittest.TestCase):
    """Unit tests for the DTLZ7 function."""

    filename = "DTLZ7.csv"

    def get_fn(self) -> benchmark.DTLZ7:
        return benchmark.DTLZ7(5)
