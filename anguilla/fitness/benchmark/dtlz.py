"""This module contains implementations of the DTLZ family of benchmark \
functions."""
import math

import numpy as np

from anguilla.fitness.base import AbstractObjectiveFunction
from anguilla.fitness.constraints import BoxConstraintsHandler


class DTLZ(AbstractObjectiveFunction):
    """Common functionality for the DTLZ family of functions."""

    def __init__(
        self,
        n_dimensions: int,
        n_objectives: int = 2,
        rng: np.random.Generator = None,
    ) -> None:
        super().__init__(n_dimensions, n_objectives, rng)

    def _pre_update_n_dimensions(self, n_dimensions: int) -> None:
        if n_dimensions < 1 or n_dimensions < self._n_objectives:
            raise ValueError("Invalid dimensions")

    def _post_update_n_dimensions(self) -> None:
        self._constraints_handler = BoxConstraintsHandler(
            self._n_dimensions, (0.0, 1.0)
        )
        self._k = self._n_dimensions - self._n_objectives + 1

    def _pre_update_n_objectives(self, n_objectives: int) -> None:
        if self._n_dimensions < n_objectives:
            raise ValueError("Invalid number of objectives")

    def _post_update_n_objectives(self) -> None:
        self._k = self._n_dimensions - self._n_objectives + 1


class DTLZ1(DTLZ):
    """The DTLZ1 multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 828 of :cite:`2002:smoo-tp`
    and follows the reference implementation in :cite:`2008:shark`.

    Learn more about this function `here <https://bit.ly/39Zn9hO>`_.
    """

    @property
    def name(self) -> str:
        return "DTLZ1"

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_single_evaluation(x)
        tmp = x[self._n_dimensions - self._k :] - 0.5
        g = 100.0 * (self._k + np.sum(tmp * tmp - np.cos(20.0 * np.pi * tmp)))
        value = np.repeat(0.5 * (1.0 + g), self._n_objectives)
        for i in range(self._n_objectives):
            for j in range(self._n_objectives - i - 1):
                value[i] *= x[j]
            if i > 0:
                value[i] *= 1.0 - x[self._n_objectives - i - 1]
        return value

    def evaluate_multiple(self, xs: np.ndarray) -> np.ndarray:
        xs = self._pre_multiple_evaluation(xs)
        tmp = xs[:, self._n_dimensions - self._k :] - 0.5
        g = 100.0 * (
            self._k + np.sum(tmp * tmp - np.cos(20.0 * np.pi * tmp), axis=1)
        )
        values = np.repeat(0.5 * (1.0 + g), self._n_objectives).reshape(
            (len(xs), self._n_objectives)
        )
        values[:, :-1] *= np.cumprod(xs[:, : self._n_objectives - 1], axis=1)[
            :, -1::-1
        ]
        values[:, 1:] *= 1.0 - xs[:, -self._k - 1 :: -1]
        return values if len(values) > 1 else values[0]


class DTLZ2(DTLZ):
    """The DTLZ1 multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 828 of :cite:`2002:smoo-tp`
    and follows the reference implementation in :cite:`2008:shark`.

    Learn more about this function `here <https://bit.ly/3gBTzAf>`_.
    """

    @property
    def name(self) -> str:
        return "DTLZ2"

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_single_evaluation(x)
        tmp = x[self._n_dimensions - self._k :] - 0.5
        g = np.sum(tmp * tmp)
        half_pi = 0.5 * math.pi
        value = np.repeat(1.0 + g, self._n_objectives)
        for i in range(self._n_objectives):
            for j in range(self._n_objectives - i - 1):
                value[i] *= math.cos(x[j] * half_pi)
            if i > 0:
                value[i] *= math.sin(x[self._n_objectives - i - 1] * half_pi)
        return value
