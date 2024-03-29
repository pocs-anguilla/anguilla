"""Testsuite for the :py:mod:`fitness.benchmark` package."""
import dataclasses
import pathlib
import unittest
from typing import Any, Optional, Tuple

import numpy as np

from anguilla.fitness.base import ObjectiveFunction
from anguilla.fitness.constraints import BoxConstraintsHandler
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
    n_dimensions = int(end_point - start_point)
    n_objectives = int(end_fitness - start_fitness)
    point = np.reshape(
        row[int(start_point) : int(end_point)], (-1, n_dimensions)
    )
    fitness = np.reshape(
        row[int(start_fitness) : int(end_fitness)], (-1, n_objectives)
    )
    return Samples(point, fitness, n_dimensions, n_objectives)


def get_samples(rows: np.ndarray) -> Samples:
    # Assumes all samples in the given rows have the same number of
    # dimensions and objectives
    start_point, end_point, start_fitness, end_fitness = rows[0, 0:4]
    points = rows[:, int(start_point) : int(end_point)]
    fitness = rows[:, int(start_fitness) : int(end_fitness)]
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
            "Evaluation count after call with single point failed, got {}.".format(
                fn.evaluation_count
            ),
        )
        fn(fn.random_points(11))
        self.assertTrue(
            fn.evaluation_count == 12,
            "Evaluation count after call with multiple points failed, got {}.".format(
                fn.evaluation_count
            ),
        )

    def test_output_shape(self):
        """Test output shape depending on the input."""
        n_points = 7
        n_dimensions = 3
        fn = self.get_fn((n_dimensions,))
        assert fn.n_dimensions == n_dimensions
        point = fn.random_points(1)
        points = fn.random_points(n_points)
        expected_single = (1, fn.n_objectives)
        expected_multiple = (n_points, fn.n_objectives)

        value = fn(point)
        computed = value.shape
        self.assertTrue(
            computed == expected_single,
            "__call__, got: {}, expected: {}".format(
                computed,
                expected_single,
            ),
        )

        value = fn(points)
        computed = value.shape
        self.assertTrue(
            computed == expected_multiple,
            "__call__, multiple points, "
            "got: {}, expected: {}".format(
                computed,
                expected_multiple,
            ),
        )

        _, value = fn.evaluate_with_penalty(point)
        computed = value.shape
        self.assertTrue(
            computed == expected_single,
            "evaluate_with_penalty, single point, "
            "got: {}, expected: {}".format(
                computed,
                expected_single,
            ),
        )

        _, value = fn.evaluate_with_penalty(points)
        computed = value.shape
        self.assertTrue(
            computed == expected_multiple,
            "evaluate_with_penalty, multiple points, "
            "got: {}, expected: {}".format(
                computed,
                expected_multiple,
            ),
        )

        value = fn.evaluate_with_constraints(point)
        computed = value.shape
        self.assertTrue(
            computed == expected_single,
            "evaluate_with_constraints, single point, "
            "got: {}, expected: {}".format(
                computed,
                expected_single,
            ),
        )

        value = fn.evaluate_with_constraints(points)
        computed = value.shape
        self.assertTrue(
            computed == expected_multiple,
            "evaluate_with_constraints, multiple points, "
            "got: {}, expected: {}".format(
                computed,
                expected_multiple,
            ),
        )

    def test_pareto_front(self):
        """Test that the Pareto front method does not throw any errors."""
        fn = self.get_fn()
        if fn.has_known_pareto_front:
            fn.n_dimensions = 2
            fn.n_objectives = 2
            fn.pareto_front()


class BaseTestFunctionSimple(BaseTestFunction):
    """Test additional properties of the function implementation."""


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
                np.allclose(fn.evaluate(sample.points), sample.fitness),
                f"Values don't match (row {i})",
            )

    def test_multiple(self):
        fn = self.get_fn()
        samples = get_samples(self.data[5:])
        fn.n_dimensions = samples.n_dimensions
        fn.n_objectives = samples.n_objectives
        result = fn.evaluate(samples.points)
        self.assertTrue(
            np.allclose(result, samples.fitness),
            "Got: {}, expected: {}".format(result, samples.fitness),
        )

    def test_call(self):
        fn = self.get_fn()
        samples = get_samples(self.data[5:])
        fn.n_dimensions = samples.n_dimensions
        fn.n_objectives = samples.n_objectives
        self.assertTrue(
            np.allclose(fn(samples.points), samples.fitness),
            "Call failed.",
        )


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


class TestIHR1Rotated(BaseTestFunctionSimple, unittest.TestCase):
    """Unit tests for the IHR1 function (with rotation matrix)."""

    fn_cls = benchmark.IHR1
    fn_kwargs = {"rotate": True}


class TestIHR2(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the IHR2 function."""

    filename = "IHR2.csv"
    fn_cls = benchmark.IHR2
    fn_kwargs = {"rotate": False}


class TestIHR2Rotated(BaseTestFunctionSimple, unittest.TestCase):
    """Unit tests for the IHR2 function (with rotation matrix)."""

    fn_cls = benchmark.IHR2
    fn_kwargs = {"rotate": True}


class TestIHR3(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the IHR3 function."""

    filename = "IHR3.csv"
    fn_cls = benchmark.IHR3
    fn_kwargs = {"rotate": False}


class TestIHR3Rotated(BaseTestFunctionSimple, unittest.TestCase):
    """Unit tests for the IHR3 function (with rotation matrix)."""

    fn_cls = benchmark.IHR3
    fn_kwargs = {"rotate": True}


class TestIHR4(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the IHR4 function."""

    filename = "IHR4.csv"
    fn_cls = benchmark.IHR4
    fn_kwargs = {"rotate": False}


class TestIHR4Rotated(BaseTestFunctionSimple, unittest.TestCase):
    """Unit tests for the IHR4 function (with rotation matrix)."""

    fn_cls = benchmark.IHR4
    fn_kwargs = {"rotate": True}


class TestIHR6(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the IHR6 function."""

    filename = "IHR6.csv"
    fn_cls = benchmark.IHR6
    fn_kwargs = {"rotate": False}


class TestIHR6Rotated(BaseTestFunctionSimple, unittest.TestCase):
    """Unit tests for the IHR6 function (with rotation matrix)."""

    fn_cls = benchmark.IHR6
    fn_kwargs = {"rotate": True}


class TestELLI1(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the ELLI1 function."""

    filename = "ELLI1.csv"
    fn_cls = benchmark.ELLI1
    fn_kwargs = {"rotate": False}


class TestELLI1Rotated(BaseTestFunctionSimple, unittest.TestCase):
    """Unit tests for the ELLI1 function (with rotation matrix)."""

    fn_cls = benchmark.ELLI1
    fn_kwargs = {"rotate": True}


class TestELLI2(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the ELLI2 function."""

    filename = "ELLI2.csv"
    fn_cls = benchmark.ELLI2
    fn_kwargs = {"rotate": False}


class TestELLI2Rotated(BaseTestFunctionSimple, unittest.TestCase):
    """Unit tests for the ELLI2 function (with rotation matrix)."""

    fn_cls = benchmark.ELLI2
    fn_kwargs = {"rotate": True}


class TestGELLI(BaseTestFunctionSimple, unittest.TestCase):
    fn_cls = benchmark.GELLI


class TestCIGTAB1(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the CIGTAB1 function."""

    filename = "CIGTAB1.csv"
    fn_cls = benchmark.CIGTAB1
    fn_kwargs = {"rotate": False, "a": 1e6}


class TestCIGTAB1Rotated(BaseTestFunctionSimple, unittest.TestCase):
    """Unit tests for the CIGTAB1 function (with rotation matrix)."""

    fn_cls = benchmark.CIGTAB1
    fn_kwargs = {"rotate": True}


class TestCIGTAB2(BaseTestFunctionWithSamples, unittest.TestCase):
    """Unit tests for the CIGTAB2 function."""

    filename = "CIGTAB2.csv"
    fn_cls = benchmark.CIGTAB2
    fn_kwargs = {"rotate": False, "a": 1e6}


class TestCIGTAB2Rotated(BaseTestFunctionSimple, unittest.TestCase):
    """Unit tests for the CIGTAB2 function (with rotation matrix)."""

    fn_cls = benchmark.CIGTAB2
    fn_kwargs = {"rotate": True}


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


class TestConstrainedEvaluation(unittest.TestCase):
    class ContrainedIdentity(ObjectiveFunction):
        def __init__(self) -> None:
            super().__init__(n_dimensions=3, n_objectives=3)

        @property
        def name(self):
            return "constrained identity"

        def evaluate(self, xs: np.ndarray, count: bool = True):
            return np.copy(xs)

        def _post_update_n_dimensions(self) -> None:
            lower_bound = np.array([-3.0, -4.0, -5.0])
            upper_bound = np.array([5.0, 4.0, 3.0])
            self._constraints_handler = BoxConstraintsHandler(
                self._n_dimensions, (lower_bound, upper_bound)
            )

    def test_nonzero_penalty(self):
        """Test evaluation of point that should be penalized."""
        penalty = 1e-6
        fn = TestConstrainedEvaluation.ContrainedIdentity()
        x = np.array([[6.0, 1.0, -6.0]])
        x_feasible = np.array([[5.0, 1.0, -5.0]])
        y_expected = x_feasible + penalty
        y_constrained, y_penalized = fn.evaluate_with_penalty(
            x, penalty=penalty
        )
        self.assertTrue(
            np.allclose(y_constrained, x_feasible),
            f"Constrained fitness, got: {y_constrained}, expected: {x_feasible}",
        )
        self.assertTrue(
            np.allclose(y_expected, y_penalized),
            f"Penalized fitness, got: {y_penalized}, expected: {y_expected}",
        )
        y_constrained_nopenalty = fn.evaluate_with_constraints(x)
        self.assertTrue(
            np.allclose(y_constrained_nopenalty, x_feasible),
            "Constrained fitness (no penalty), got: {}, expected: {}".format(
                y_constrained_nopenalty, x_feasible
            ),
        )

    def test_zero_penalty(self):
        """Test evaluation of point within constraints."""
        penalty = 1e-6
        fn = TestConstrainedEvaluation.ContrainedIdentity()
        x = np.array([[1.0, -2.0, 3.0]])
        y_constrained, y_penalized = fn.evaluate_with_penalty(
            x, penalty=penalty
        )
        self.assertTrue(
            np.allclose(y_constrained, x),
            f"Constrained fitness, got: {y_constrained}, expected: {x}",
        )
        self.assertTrue(
            np.allclose(y_penalized, x),
            f"Penalized fitness, got: {y_penalized}, expected: {x}",
        )
        y_constrained_nopenalty = fn.evaluate_with_constraints(x)
        self.assertTrue(
            np.allclose(y_constrained_nopenalty, x),
            "Constrained fitness (no penalty), got: {}, expected: {}".format(
                y_constrained_nopenalty, x
            ),
        )
