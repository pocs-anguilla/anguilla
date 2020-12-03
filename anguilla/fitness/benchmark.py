"""This module contains benchmark fitness/objective functions."""

import math
import typing

import numpy as np

from anguilla.fitness.base import AbstractObjectiveFunction
from anguilla.fitness.constraints import BoxConstraintsHandler
from anguilla.util import random_orthogonal_matrix


class Sphere(AbstractObjectiveFunction):
    """The Sphere single-objective function.

    Notes
    -----
    `Learn more <https://www.sfu.ca/~ssurjano/spheref.html>`_ about this \
    function at :cite:`2013:simulationlib`.
    """

    @property
    def name(self) -> str:
        return "Sphere"

    def evaluate(self, x: np.ndarray) -> typing.Union[float, np.ndarray]:
        if len(x.shape) > 1:
            axis = 1
            self._evaluation_count += x.shape[0]
        else:
            axis = 0
            self._evaluation_count += 1
        return np.sum(x * x, axis=axis)

    evaluate_multiple = evaluate


class SumSquares(AbstractObjectiveFunction):
    """The Axis Parallel Hyper-Ellipsoid single-objective function.

    Notes
    -----
    `Learn more <https://www.sfu.ca/~ssurjano/sumsqu.html>`_ about this \
    function at :cite:`2013:simulationlib`.
    """

    @property
    def name(self) -> str:
        return "SumSquares"

    def evaluate(self, x: np.ndarray) -> typing.Union[float, np.ndarray]:
        if len(x.shape) > 1:
            axis = 1
            self._evaluation_count += x.shape[0]
        else:
            axis = 0
            self._evaluation_count += 1
        d = x.shape[axis]
        return np.sum(np.arange(1, d + 1) * (x * x), axis=axis)

    evaluate_multiple = evaluate


class FON(AbstractObjectiveFunction):
    """The FON multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 14 of :cite:`2007:mo-cma-es`.
    """

    _n_sqrt: float

    def __init__(
        self, n_dimensions: int, rng: np.random.Generator = None
    ) -> None:
        super().__init__(n_dimensions, rng)
        self._n_objectives = 2

    @property
    def name(self) -> str:
        return "FON"

    def _handle_dimensions_update(self) -> None:
        self._constraints_handler = BoxConstraintsHandler(
            self._n_dimensions, (-4.0, 4.0)
        )
        self._n_sqrt = math.sqrt(self._n_dimensions)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        self._validate_point_shape(x.shape)
        self._evaluation_count += 1
        value = np.zeros(self._n_objectives)
        tmp1 = x - self._n_sqrt
        tmp2 = x + self._n_sqrt
        value[0] = 1.0 - math.exp(-np.sum(tmp1 * tmp1))
        value[1] = 1.0 - math.exp(-np.sum(tmp2 * tmp2))
        return value

    def evaluate_multiple(self, x: np.ndarray) -> np.ndarray:
        axis = self._validate_points_shape(x.shape)
        values = np.zeros((x.shape[0], self._n_objectives))
        self._evaluation_count += x.shape[0]
        tmp1 = x - self._n_sqrt
        tmp2 = x + self._n_sqrt
        values[:, 0] = 1.0 - np.exp(-np.sum(tmp1 * tmp1, axis=axis))
        values[:, 1] = 1.0 - np.exp(-np.sum(tmp2 * tmp2, axis=axis))
        return values


class ZDT4P(AbstractObjectiveFunction):
    """The ZDT4' multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 16 of :cite:`2007:mo-cma-es`.
    """

    def __init__(
        self, n_dimensions: int, rng: np.random.Generator = None
    ) -> None:
        super().__init__(n_dimensions, rng)
        self._n_objectives = 2

    @property
    def name(self) -> str:
        return "ZDT4'"

    def _handle_dimensions_update(self) -> None:
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

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        self._validate_point_shape(x.shape)
        self._evaluation_count += 1
        # First objective
        value = np.array([x[0], 0.0])
        # Second objective
        # y = Ox
        y = self._rotation_matrix @ x
        # g(y) = (...)
        value[1] = (10.0 * (self._n_dimensions - 1)) + np.sum(
            y[1:] * y[1:] - 10.0 * np.cos(4.0 * np.pi * y[1:])
        )
        # f2(x) = (...)
        value[1] *= 1.0 - np.sqrt(x[0] / value[1])
        return value

    def evaluate_multiple(self, x: np.ndarray) -> np.ndarray:
        # Setup
        axis = self._validate_points_shape(x.shape)
        self._evaluation_count += x.shape[0]
        if axis == 0:
            x = x.reshape(1, x.shape[0])
        shape = (x.shape[0], self._n_objectives)
        values = np.zeros(shape)
        # First objective
        values[:, 0] = x[:, 0]
        # Second objective
        n = self._n_dimensions
        # y = Ox
        y = np.zeros_like(x)
        for i in range(x.shape[0]):
            y[i] = self._rotation_matrix @ x[i]
        # g(y) = (...)
        values[:, 1] = (10.0 * (n - 1)) + np.sum(
            y[:, 1:] * y[:, 1:] - 10.0 * np.cos(4.0 * np.pi * y[:, 1:]),
            axis=1,
        )
        # f2(x) = (...)
        values[:, 1] *= 1.0 - np.sqrt(values[:, 0] / values[:, 1])
        return values if axis != 0 else values[0]
