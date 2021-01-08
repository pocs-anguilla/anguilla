"""This module contains implementations of the IHR family of benchmark \
functions."""

import math

import numpy as np

from anguilla.fitness.base import ObjectiveFunction, BoundsTuple
from anguilla.fitness.constraints import BoxConstraintsHandler
from anguilla.util import random_orthogonal_matrix


def _h(x: float, n: float) -> float:
    """Compute an auxiliary function for some benchmark functions.

    Notes
    -----
    Implements the :math:`h` auxiliary function as defined in \
    p. 16 of :cite:`2007:mo-cma-es`.
    """
    return 1.0 / (1.0 + math.exp(-x / math.sqrt(n)))


def _h_f(x: float, y1_abs: float, ymax: float) -> float:
    """Compute an auxiliary function for some benchmark functions.

    Notes
    -----
    Implements the :math:`h_f` auxiliary function as defined in \
    p. 16 of :cite:`2007:mo-cma-es`.
    """
    if y1_abs < ymax:
        return x
    return 1.0 + y1_abs


def _h_g(x: np.ndarray) -> np.ndarray:
    """Compute an auxiliary function for some benchmark functions.

    Notes
    -----
    Implements the :math:`h_g` auxiliary function as defined in \
    p. 16 of :cite:`2007:mo-cma-es`.
    """
    return (x * x) / (np.abs(x) + 0.1)


def _ihr_g(y: np.ndarray, n: int) -> float:
    """Compute g(y) for the IHR1, IHR2 and IHR3 benchmark functions."""
    return 1.0 + 9.0 * (np.sum(_h_g(y[1:])) / (n - 1))


class IHR(ObjectiveFunction):
    """Common functionality for the IHR family of functions.

    Parameters
    ----------
    n_dimensions
        Dimensionality of the search space
    rng: optional
        A random number generator.
    """

    _bounds: BoundsTuple
    _rotation_matrix: np.ndarray
    _y_max: float
    _rotate: bool

    def __init__(
        self,
        n_dimensions: int = 2,
        rng: np.random.Generator = None,
        rotate: bool = True,
    ) -> None:
        self._rotate = rotate
        super().__init__(n_dimensions, 2, rng)

    def _post_update_n_dimensions(self) -> None:
        self._rotation_matrix = (
            random_orthogonal_matrix(self._n_dimensions, self._rng)
            if self._rotate
            else np.eye(self._n_dimensions)
        )
        self._y_max = 1.0 / np.linalg.norm(self._rotation_matrix[0], np.inf)
        self._constraints_handler = BoxConstraintsHandler(
            self._n_dimensions, self._bounds
        )


class IHR1(IHR):
    """The IHR1 multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 16 of :cite:`2007:mo-cma-es`.
    """

    _bounds = (-1.0, 1.0)

    @property
    def name(self) -> str:
        return "IHR1"

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_evaluate_single(x)
        n = self._n_dimensions
        y = self._rotation_matrix @ x
        value = np.array([abs(y[0]), 0.0])
        h = _h(y[0], n)
        g = _ihr_g(y, n)
        value[1] = g * _h_f(1.0 - math.sqrt(h / g), y[0], self._y_max)
        return value


class IHR2(IHR):
    """The IHR2 multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 16 of :cite:`2007:mo-cma-es`.
    """

    _bounds = (-1.0, 1.0)

    @property
    def name(self) -> str:
        return "IHR2"

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_evaluate_single(x)
        n = self._n_dimensions
        y = self._rotation_matrix @ x
        value = np.array([abs(y[0]), 0.0])
        g = _ihr_g(y, n)
        tmp = y[0] / g
        value[1] = g * _h_f(1.0 - tmp * tmp, value[0], self._y_max)
        return value


class IHR3(IHR):
    """The IHR3 multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 16 of :cite:`2007:mo-cma-es`.
    """

    _bounds = (-1.0, 1.0)

    @property
    def name(self) -> str:
        return "IHR3"

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_evaluate_single(x)
        n = self._n_dimensions
        y = self._rotation_matrix @ x
        value = np.array([abs(y[0]), 0.0])
        h = _h(y[0], n)
        g = _ihr_g(y, n)
        tmp1 = h / g
        tmp2 = tmp1 * math.sin(10.0 * math.pi * y[0])
        value[1] = g * _h_f(
            1.0 - math.sqrt(tmp1) - tmp2, value[0], self._y_max
        )
        return value


class IHR4(IHR):
    """The IHR4 multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 16 of :cite:`2007:mo-cma-es`.
    """

    _bounds = (-5.0, 5.0)

    @property
    def name(self) -> str:
        return "IHR4"

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_evaluate_single(x)
        n = self._n_dimensions
        y = self._rotation_matrix @ x
        value = np.array([abs(y[0]), 0.0])
        h = _h(y[0], n)
        g = (
            1.0
            + 10.0 * (n - 1)
            + np.sum(y[1:] * y[1:] - 10.0 * np.cos(4.0 * np.pi * y[1:]))
        )
        value[1] = g * _h_f(1.0 - math.sqrt(h / g), value[0], self._y_max)
        return value


class IHR6(IHR):
    """The IHR6 multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 16 of :cite:`2007:mo-cma-es`.
    """

    _bounds = (-5.0, 5.0)

    @property
    def name(self) -> str:
        return "IHR6"

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_evaluate_single(x)
        n = self._n_dimensions
        y = self._rotation_matrix @ x
        value = np.zeros(2)
        value[0] = 1.0 - math.exp(-4.0 * abs(y[0])) * math.pow(
            math.sin(6.0 * math.pi * y[0]), 6
        )
        g = 1.0 + 9.0 * math.pow(np.sum(_h_g(y[1:])) / (n - 1), 0.25)
        tmp = value[0] / g
        value[1] = g * _h_f(1.0 - (tmp * tmp), abs(y[0]), self._y_max)
        return value
