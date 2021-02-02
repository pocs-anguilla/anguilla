"""This module contains implementations of the ZDT family of benchmark \
functions."""
import math

import numpy as np

from anguilla.fitness.base import ObjectiveFunction
from anguilla.fitness.constraints import BoxConstraintsHandler
from anguilla.util import random_orthogonal_matrix


class ZDT1(ObjectiveFunction):
    """The ZDT1 multi-objective box-constrained function from the ZDT benchmark.

    Notes
    -----
    Implements the function as defined in p. 14 of :cite:`2007:mo-cma-es` and \
    the true Pareto front as in `this page <https://bit.ly/2XdDQym>`_.
    """

    _has_known_pareto_front = True

    def __init__(
        self, n_dimensions: int = 2, rng: np.random.Generator = None
    ) -> None:
        super().__init__(n_dimensions, 2, rng)

    @property
    def name(self) -> str:
        return "ZDT1"

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_evaluate_single(x)
        value = np.empty(self._n_objectives)
        value[0] = x[0]
        g = 1.0 + 9.0 * np.average(x[1:])
        value[1] = g * (1.0 - math.sqrt(x[0] / g))
        return value

    def evaluate_multiple(self, xs: np.ndarray) -> np.ndarray:
        xs = self._pre_evaluate_multiple(xs)
        values = np.empty((len(xs), self._n_objectives))
        values[:, 0] = xs[:, 0]
        g = 1.0 + 9.0 * np.average(xs[:, 1:], axis=1)
        values[:, 1] = g * (1.0 - np.sqrt(xs[:, 0] / g))
        return values if len(xs) > 1 else values[0]

    def pareto_front(self, num=50) -> np.ndarray:
        y1 = np.linspace(0.0, 1.0, num, True)
        y2 = 1.0 - np.sqrt(y1)
        return np.vstack((y1, y2))

    def _post_update_n_dimensions(self) -> None:
        self._constraints_handler = BoxConstraintsHandler(
            self._n_dimensions, (0.0, 1.0)
        )


class ZDT2(ObjectiveFunction):
    """The ZDT2 multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 14 of :cite:`2007:mo-cma-es` and \
    the true Pareto front as in `this page <https://bit.ly/38hfhXN>`_.
    """

    _has_known_pareto_front = True

    def __init__(
        self, n_dimensions: int = 2, rng: np.random.Generator = None
    ) -> None:
        super().__init__(n_dimensions, 2, rng)

    @property
    def name(self) -> str:
        return "ZDT2"

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_evaluate_single(x)
        value = np.empty(self._n_objectives)
        value[0] = x[0]
        g = 1.0 + 9.0 * np.average(x[1:])
        tmp = x[0] / g
        value[1] = g * (1.0 - tmp * tmp)
        return value

    def evaluate_multiple(self, xs: np.ndarray) -> np.ndarray:
        xs = self._pre_evaluate_multiple(xs)
        values = np.empty((len(xs), self._n_objectives))
        values[:, 0] = xs[:, 0]
        g = 1.0 + 9.0 * np.average(xs[:, 1:], axis=1)
        tmp = xs[:, 0] / g
        values[:, 1] = g * (1.0 - tmp * tmp)
        return values if len(xs) > 1 else values[0]

    def pareto_front(self, num=50) -> np.ndarray:
        y1 = np.linspace(0.0, 1.0, num, True)
        y2 = 1.0 - y1 * y1
        return np.vstack((y1, y2))

    def _post_update_n_dimensions(self) -> None:
        self._constraints_handler = BoxConstraintsHandler(
            self._n_dimensions, (0.0, 1.0)
        )


class ZDT3(ObjectiveFunction):
    """The ZDT3 multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 14 of :cite:`2007:mo-cma-es` and \
    the true Pareto front as in `this page <https://bit.ly/2LsWeRm>`_.
    """

    _has_known_pareto_front = True
    _has_continuous_pareto_front = False

    def __init__(
        self, n_dimensions: int = 2, rng: np.random.Generator = None
    ) -> None:
        super().__init__(n_dimensions, 2, rng)

    @property
    def name(self) -> str:
        return "ZDT3"

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_evaluate_single(x)
        value = np.empty(self._n_objectives)
        value[0] = x[0]
        g = 1.0 + 9.0 * np.average(x[1:])
        tmp = x[0] / g
        value[1] = g * (
            1.0 - math.sqrt(tmp) - tmp * math.sin(10.0 * math.pi * x[0])
        )
        return value

    def evaluate_multiple(self, xs: np.ndarray) -> np.ndarray:
        xs = self._pre_evaluate_multiple(xs)
        values = np.empty((len(xs), self._n_objectives))
        values[:, 0] = xs[:, 0]
        g = 1.0 + 9.0 * np.average(xs[:, 1:], axis=1)
        tmp = xs[:, 0] / g
        values[:, 1] = g * (
            1.0 - np.sqrt(tmp) - tmp * np.sin(10.0 * np.pi * xs[:, 0])
        )
        return values if len(xs) > 1 else values[0]

    def pareto_front(self, num=50) -> np.ndarray:
        size = num // 5
        y1 = np.empty(num)
        y1[:size] = np.linspace(0.0, 0.0830015349, size, True)
        y1[size : size * 2] = np.linspace(
            0.1822287280, 0.2577623634, size, True
        )
        y1[size * 2 : size * 3] = np.linspace(
            0.4093136748, 0.4538821041, size, True
        )
        y1[size * 3 : size * 4] = np.linspace(
            0.6183967944, 0.6525117038, size, True
        )
        y1[size * 4 :] = np.linspace(
            0.8233317983, 0.8518328654, size + num % 5, True
        )
        y2 = 1.0 - np.sqrt(y1) - y1 * np.sin(10.0 * np.pi * y1)
        return np.vstack((y1, y2))

    def _post_update_n_dimensions(self) -> None:
        self._constraints_handler = BoxConstraintsHandler(
            self._n_dimensions, (0.0, 1.0)
        )


class ZDT4(ObjectiveFunction):
    """The ZDT4 multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 14 of :cite:`2007:mo-cma-es` \
    and the true Pareto front as in `this page <https://bit.ly/2Ls4fWk>`_.
    """

    _has_known_pareto_front = True

    def __init__(
        self, n_dimensions: int = 2, rng: np.random.Generator = None
    ) -> None:
        super().__init__(n_dimensions, 2, rng)

    @property
    def name(self) -> str:
        return "ZDT4"

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_evaluate_single(x)
        value = np.empty(self._n_objectives)
        value[0] = x[0]
        g = (
            1.0
            + 10.0 * (self._n_dimensions - 1)
            + np.sum(x[1:] * x[1:] - 10.0 * np.cos(4.0 * np.pi * x[1:]))
        )
        value[1] = g * (1.0 - math.sqrt(x[0] / g))
        return value

    def evaluate_multiple(self, xs: np.ndarray) -> np.ndarray:
        xs = self._pre_evaluate_multiple(xs)
        values = np.empty((len(xs), self._n_objectives))
        values[:, 0] = xs[:, 0]
        g = (
            1.0
            + 10.0 * (self._n_dimensions - 1)
            + np.sum(
                xs[:, 1:] * xs[:, 1:] - 10.0 * np.cos(4.0 * np.pi * xs[:, 1:]),
                axis=1,
            )
        )
        values[:, 1] = g * (1.0 - np.sqrt(xs[:, 0] / g))
        return values if len(xs) > 1 else values[0]

    def pareto_front(self, num=50) -> np.ndarray:
        y1 = np.linspace(0.0, 1.0, num, True)
        y2 = 1.0 - np.sqrt(y1)
        return np.vstack((y1, y2))

    def _post_update_n_dimensions(self) -> None:
        n = self._n_dimensions
        # Update box constraints
        lower_bounds = np.repeat(-5.0, n)
        lower_bounds[0] = 0.0
        upper_bounds = np.repeat(5.0, n)
        upper_bounds[0] = 1.0
        self._constraints_handler = BoxConstraintsHandler(
            self._n_dimensions, (lower_bounds, upper_bounds)
        )


class ZDT4P(ObjectiveFunction):
    """The ZDT4' multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 16 of :cite:`2007:mo-cma-es`.
    """

    def __init__(
        self, n_dimensions: int = 2, rng: np.random.Generator = None
    ) -> None:
        super().__init__(n_dimensions, 2, rng)

    @property
    def name(self) -> str:
        return "ZDT4'"

    def _post_update_n_dimensions(self) -> None:
        n = self._n_dimensions
        # Update box constraints
        lower_bounds = np.repeat(-5.0, n)
        lower_bounds[0] = 0.0
        upper_bounds = np.repeat(5.0, n)
        upper_bounds[0] = 1.0
        self._constraints_handler = BoxConstraintsHandler(
            self._n_dimensions, (lower_bounds, upper_bounds)
        )
        # Create the random orthogonal (and restricted) rotation matrix
        self._rotation_matrix = np.zeros((n, n))
        self._rotation_matrix[0, 0] = 1.0
        self._rotation_matrix[1:, 1:] = random_orthogonal_matrix(
            n - 1, self._rng
        )

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_evaluate_single(x)
        # First objective
        value = np.array([x[0], 0.0])
        # Second objective
        y = self._rotation_matrix @ x

        value[1] = (
            1.0
            + (10.0 * float(self._n_dimensions - 1))
            + np.sum(y[1:] * y[1:] - 10.0 * np.cos(4.0 * np.pi * y[1:]))
        )

        value[1] *= 1.0 - np.sqrt(x[0] / value[1])
        return value

    def evaluate_multiple(self, xs: np.ndarray) -> np.ndarray:
        # Setup
        xs = self._pre_evaluate_multiple(xs)
        n_points = len(xs)
        shape = (n_points, self._n_objectives)
        values = np.zeros(shape)
        # First objective
        values[:, 0] = xs[:, 0]
        # Second objective
        n = self._n_dimensions
        y = np.zeros_like(xs)
        for i in range(n_points):
            y[i] = self._rotation_matrix @ xs[i]

        values[:, 1] = (
            1.0
            + (10.0 * float(n - 1))
            + np.sum(
                y[:, 1:] * y[:, 1:] - 10.0 * np.cos(4.0 * np.pi * y[:, 1:]),
                axis=1,
            )
        )

        values[:, 1] *= 1.0 - np.sqrt(values[:, 0] / values[:, 1])
        return values if len(xs.shape) > 1 else values[0]


class ZDT6(ObjectiveFunction):
    """The ZDT6 multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 14 of :cite:`2007:mo-cma-es` and \
    the Pareto front as in `this page <https://bit.ly/38hkXAZ>`_.
    """

    _has_known_pareto_front = True

    def __init__(
        self, n_dimensions: int = 2, rng: np.random.Generator = None
    ) -> None:
        super().__init__(n_dimensions, 2, rng)

    @property
    def name(self) -> str:
        return "ZDT6"

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_evaluate_single(x)
        value = np.empty(self._n_objectives)
        value[0] = 1.0 - math.exp(-4.0 * x[0]) * math.pow(
            math.sin(6.0 * math.pi * x[0]), 6.0
        )
        g = 1.0 + 9.0 * math.pow(np.average(x[1:]), 0.25)
        tmp = value[0] / g
        value[1] = g * (1.0 - tmp * tmp)
        return value

    def evaluate_multiple(self, xs: np.ndarray) -> np.ndarray:
        xs = self._pre_evaluate_multiple(xs)
        values = np.empty((len(xs), self._n_objectives))
        values[:, 0] = 1.0 - np.exp(-4.0 * xs[:, 0]) * np.power(
            np.sin(6.0 * np.pi * xs[:, 0]), 6.0
        )
        g = 1.0 + 9.0 * np.power(np.average(xs[:, 1:], axis=1), 0.25)
        tmp = values[:, 0] / g
        values[:, 1] = g * (1.0 - tmp * tmp)
        return values if len(xs) > 1 else values[0]

    def _post_update_n_dimensions(self) -> None:
        self._constraints_handler = BoxConstraintsHandler(
            self._n_dimensions, (0.0, 1.0)
        )

    def pareto_front(self, num=50) -> np.ndarray:
        y1 = np.linspace(0.2807753191, 1.0, endpoint=True, num=num)
        y2 = 1.0 - y1 * y1
        return y1, y2


__all__ = [
    "ZDT1",
    "ZDT2",
    "ZDT3",
    "ZDT4",
    "ZDT4P",
    "ZDT6",
]
