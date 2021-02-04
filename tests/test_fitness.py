"""Testsuite for the :py:mod:`fitness.benchmark` package."""
import dataclasses
import pathlib
import unittest
from typing import Any, Optional, Tuple

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
    """Base class to test benchmark objective functions.

    Notes
    -----
    Subclasses use the inheritance approach suggested \
    `here <https://stackoverflow.com/a/52261358>`_.
    """

    # We include this property so that static analyzers don't complain.
    # Concrete subclasses also inherit from unittest.TestCase
    # which has an implementation of this method.
    assertTrue: Any

    fn_cls: ObjectiveFunction
    fn_args: Optional[Tuple[Any]] = None
    fn_kwargs: Optional[dict] = None
    # Some functions should start with these args always
    # and cannot be overrided with custom values
    fn_args_mandatory: Optional[Tuple[Any]] = None

    def get_fn(
        self,
        custom_args: Optional[Tuple] = None,
        custom_kwargs: Optional[dict] = None,
    ) -> ObjectiveFunction:
        args = ()
        if self.fn_args is not None:
            args = self.fn_args
        if custom_args is not None:
            args = custom_args

        if self.fn_args_mandatory is not None:
            args = self.fn_args_mandatory + args

        kwargs = {}
        if self.fn_kwargs is not None:
            kwargs = self.fn_kwargs
        if custom_kwargs is not None:
            kwargs = custom_kwargs

        return self.fn_cls(*args, **kwargs)

    def test_initialization(self):
        """Test that the function object is initialized without errors"""
        fn = self.get_fn()
        rng = np.random.default_rng()
        n_dimensions = None
        n_objectives = None

        if fn.has_scalable_dimensions:
            n_dimensions = rng.integers(11, 15)

        if fn.has_scalable_objectives:
            n_objectives = rng.integers(5, 10)

        if n_dimensions is not None:
            fn1 = self.get_fn(custom_args=(n_dimensions,))
            self.assertTrue(fn1.n_dimensions == n_dimensions, "n_dimensions")

        if n_objectives is not None:
            fn2 = self.get_fn(custom_kwargs={"n_objectives": n_objectives})
            self.assertTrue(fn2.n_objectives == n_objectives, "n_objectives")

        if n_dimensions is not None and n_objectives is not None:
            fn3 = self.get_fn(custom_args=(n_dimensions, n_objectives))
            self.assertTrue(
                (
                    fn3.n_dimensions == n_dimensions
                    and fn3.n_objectives == n_objectives
                ),
                "n_dimensions and n_objectives",
            )

    def test_scale_dimensions(self):
        """Test that the dimensions can be scaled."""
        fn = self.get_fn()
        start = fn.n_dimensions
        fn.n_dimensions = start + 1
        end = fn.n_dimensions
        if fn.has_scalable_dimensions:
            self.assertTrue(start != end, "!= operator")
        else:
            self.assertTrue(start == end, "== operator")

    def test_scale_objectives(self):
        """Test that the objectives can be scaled."""
        fn = self.get_fn()
        if fn.has_scalable_dimensions:
            # Some functions with scalable objectives and dimensions
            # require that n_objectives <= n_dimensions
            fn = self.get_fn()
            start = fn.n_objectives
            fn.n_dimensions = fn.n_objectives + 3
            fn.n_objectives = start + 1
            end = fn.n_objectives
            if fn.has_scalable_objectives:
                self.assertTrue(start != end, "!= operator")
            else:
                self.assertTrue(start == end, "== operator")
        else:
            start = fn.n_objectives
            fn.n_objectives = start + 1
            end = fn.n_objectives
            if fn.has_scalable_objectives:
                self.assertTrue(start != end, "!= operator")
            else:
                self.assertTrue(start == end, "== operator")

    def test_evaluation_count(self):
        """Test that the evaluation counter is updated correctly."""
        fn = self.get_fn()
        fn.n_dimensions = 5
        self.assertTrue(
            fn.evaluation_count == 0, "Initial evaluation count failed."
        )
        fn(fn.random_points(1))
        self.assertTrue(
            fn.evaluation_count == 1,
            "Evaluation count after single call failed, got {}.".format(
                fn.evaluation_count
            ),
        )
        fn(fn.random_points(11))
        self.assertTrue(
            fn.evaluation_count == 12,
            "Evaluation count after multiple call failed, got {}.".format(
                fn.evaluation_count
            ),
        )


class BaseTestFunctionSimple(BaseTestFunction):
    """Test additional properties of the function implementation."""

    def test_call_single_multiple(self):
        fn = self.get_fn()
        n_points = 10
        xs = fn.random_points(n_points)
        single_ys = np.empty((n_points, fn.n_objectives))
        for i in range(n_points):
            single_ys[i] = fn(xs[i])
        multiple_ys = fn(xs)
        self.assertTrue(np.allclose(single_ys, multiple_ys))


class BaseTestFunctionWithSamples(BaseTestFunction):
    """Test evaluation results against pre-computed samples."""

    filename: str

    def setUp(self):
        path = (
            pathlib.Path(__file__)
            .parent.joinpath("data/fitness")
            .joinpath(self.filename)
            .absolute()
        )
        self.data = np.genfromtxt(str(path), delimiter=",")

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


class TestSphere(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the Sphere function."""

    filename = "Sphere.csv"
    fn_cls = benchmark.Sphere


class TestRastrigin(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the Rastrigin function."""

    filename = "Rastrigin.csv"
    fn_cls = benchmark.Rastrigin
    fn_kwargs = {"rotate": False}


class TestEllipsoid(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the Ellipsoid function."""

    filename = "Ellipsoid.csv"
    fn_cls = benchmark.Ellipsoid
    fn_kwargs = {"rotate": False}


class TestFON(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the FON function."""

    filename = "FON.csv"
    fn_cls = benchmark.FON


class TestZDT1(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the ZDT1 function."""

    filename = "ZDT1.csv"
    fn_cls = benchmark.ZDT1


class TestZDT2(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the ZDT2 function."""

    filename = "ZDT2.csv"
    fn_cls = benchmark.ZDT2


class TestZDT3(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the ZDT3 function."""

    filename = "ZDT3.csv"
    fn_cls = benchmark.ZDT3


class TestZDT4(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the ZDT4 function."""

    filename = "ZDT4.csv"
    fn_cls = benchmark.ZDT4


class TestZDT6(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the ZDT6 function."""

    filename = "ZDT6.csv"
    fn_cls = benchmark.ZDT6


class TestIHR1(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the IHR1 function."""

    filename = "IHR1.csv"
    fn_cls = benchmark.IHR1
    fn_kwargs = {"rotate": False}


class TestIHR2(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the IHR2 function."""

    filename = "IHR2.csv"
    fn_cls = benchmark.IHR2
    fn_kwargs = {"rotate": False}


class TestELLI1(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the ELLI1 function."""

    filename = "ELLI1.csv"
    fn_cls = benchmark.ELLI1
    fn_kwargs = {"rotate": False}


class TestELLI2(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the ELLI2 function."""

    filename = "ELLI2.csv"
    fn_cls = benchmark.ELLI2
    fn_kwargs = {"rotate": False}


class TestGELLI(BaseTestFunctionSimple, unittest.TestCase):
    fn_cls = benchmark.GELLI


class TestCIGTAB1(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the CIGTAB1 function."""

    filename = "CIGTAB1.csv"
    fn_cls = benchmark.CIGTAB1
    fn_kwargs = {"rotate": False}


class TestCIGTAB2(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the CIGTAB2 function."""

    filename = "CIGTAB2.csv"
    fn_cls = benchmark.CIGTAB2
    fn_kwargs = {"rotate": False}


class TestDTLZ1(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the DTLZ1 function."""

    filename = "DTLZ1.csv"
    fn_cls = benchmark.DTLZ1
    fn_args = (5,)


class TestDTLZ2(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the DTLZ2 function."""

    filename = "DTLZ2.csv"
    fn_cls = benchmark.DTLZ2
    fn_args = (5,)


class TestDTLZ3(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the DTLZ3 function."""

    filename = "DTLZ3.csv"
    fn_cls = benchmark.DTLZ3
    fn_args = (5,)


class TestDTLZ4(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the DTLZ4 function."""

    filename = "DTLZ4.csv"
    fn_cls = benchmark.DTLZ4
    fn_args = (5,)


class TestDTLZ5(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the DTLZ5 function."""

    filename = "DTLZ5.csv"
    fn_cls = benchmark.DTLZ5
    fn_args = (5,)


class TestDTLZ6(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the DTLZ6 function."""

    filename = "DTLZ6.csv"
    fn_cls = benchmark.DTLZ6
    fn_args = (5,)


class TestDTLZ7(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the DTLZ7 function."""

    filename = "DTLZ7.csv"
    fn_cls = benchmark.DTLZ7
    fn_args = (5,)


class TestMOQ1AC(BaseTestFunctionSimple, unittest.TestCase):
    fn_cls = benchmark.MOQ
    fn_args_mandatory = ("1|C",)


class TestMOQ1NAC(BaseTestFunctionSimple, unittest.TestCase):
    fn_cls = benchmark.MOQ
    fn_args_mandatory = ("1/C",)


class TestMOQ1AJ(BaseTestFunctionSimple, unittest.TestCase):
    fn_cls = benchmark.MOQ
    fn_args_mandatory = ("1|J",)


class TestMOQ1NAJ(BaseTestFunctionSimple, unittest.TestCase):
    fn_cls = benchmark.MOQ
    fn_args_mandatory = ("1/J",)


class TestMOQ1AI(BaseTestFunctionSimple, unittest.TestCase):
    fn_cls = benchmark.MOQ
    fn_args_mandatory = ("1|I",)


class TestMOQ1NAI(BaseTestFunctionSimple, unittest.TestCase):
    fn_cls = benchmark.MOQ
    fn_args_mandatory = ("1/I",)
