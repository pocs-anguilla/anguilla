"""This module contains implementations of the DTLZ family of benchmark \
functions."""
import math

import numpy as np

from anguilla.fitness.base import ObjectiveFunction
from anguilla.fitness.constraints import BoxConstraintsHandler


class DTLZ(ObjectiveFunction):
    """Common functionality for the DTLZ family of functions."""

    _has_scalable_objectives = True
    _has_known_pareto_front = True

    def __init__(
        self,
        n_dimensions: int = 2,
        n_objectives: int = 2,
        rng: np.random.Generator = None,
    ) -> None:
        self._k = None
        super().__init__(n_dimensions, n_objectives, rng)

    @property
    def name(self) -> str:
        return "DTLZ"

    def evaluate(self, xs: np.ndarray, count: bool = True) -> np.ndarray:
        raise NotImplementedError()

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
    Implements the function as defined in p. 4 of :cite:`2002:smoo-tp`
    and follows the reference implementation in :cite:`2008:shark`.

    Implements the true Pareto front as defined in \
    `this page <https://bit.ly/3pV2hwU>`_.
    """

    @property
    def name(self) -> str:
        return "DTLZ1"

    def evaluate(self, xs: np.ndarray, count: bool = True) -> np.ndarray:
        self._pre_evaluate(xs, count=count)
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
        return values

    def pareto_front(self, num=50) -> np.ndarray:
        if self._n_objectives == 2:
            y1 = np.linspace(0.0, 0.5, num, True)
            y2 = 0.5 - y1
            return np.vstack((y1, y2))
        raise NotImplementedError()


class DTLZ2(DTLZ):
    """The DTLZ2 multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 4 of :cite:`2002:smoo-tp`
    and follows the reference implementation in :cite:`2008:shark`.

    Implements the true Pareto front as defined in \
    `this page <https://bit.ly/3olA01U>`_.
    """

    @property
    def name(self) -> str:
        return "DTLZ2"

    def evaluate(self, xs: np.ndarray, count: bool = True) -> np.ndarray:
        self._pre_evaluate(xs, count=count)
        values = np.empty(shape=(len(xs), self._n_objectives))
        for i in range(len(xs)):
            tmp = xs[i, self._n_dimensions - self._k :] - 0.5
            g = np.sum(tmp * tmp)
            half_pi = 0.5 * math.pi
            values[i, :] = np.repeat(1.0 + g, self._n_objectives)
            for a in range(self._n_objectives):
                for b in range(self._n_objectives - a - 1):
                    values[i, a] *= math.cos(xs[i, b] * half_pi)
                if a > 0:
                    values[i, a] *= math.sin(
                        xs[i, self._n_objectives - a - 1] * half_pi
                    )
        return values

    def pareto_front(self, num=50) -> np.ndarray:
        if self._n_objectives == 2:
            y1 = np.linspace(0.0, 1.0, num, True)
            y2 = np.sqrt(1.0 - y1 * y1)
            return np.vstack((y1, y2))
        raise NotImplementedError()


class DTLZ3(DTLZ):
    """The DTLZ3 multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 4 of :cite:`2002:smoo-tp`
    and follows the reference implementation in :cite:`2008:shark`.

    Implements the true Pareto front as defined in \
    `this page <https://bit.ly/3njpdEi>`_.
    """

    @property
    def name(self) -> str:
        return "DTLZ3"

    def evaluate(self, xs: np.ndarray, count: bool = True) -> np.ndarray:
        self._pre_evaluate(xs, count=count)
        values = np.empty(shape=(len(xs), self._n_objectives))
        for i in range(len(xs)):
            tmp = xs[i, self._n_dimensions - self._k :] - 0.5
            g = 100.0 * (
                self._k + np.sum(tmp * tmp - np.cos(20.0 * np.pi * tmp))
            )
            half_pi = 0.5 * math.pi
            values[i, :] = np.repeat(1.0 + g, self._n_objectives)
            for a in range(self._n_objectives):
                for b in range(self._n_objectives - a - 1):
                    values[i, a] *= math.cos(xs[i, b] * half_pi)
                if a > 0:
                    values[i, a] *= math.sin(
                        xs[i, self._n_objectives - a - 1] * half_pi
                    )
        return values

    def pareto_front(self, num=50) -> np.ndarray:
        if self._n_objectives == 2:
            y1 = np.linspace(0.0, 1.0, num, True)
            y2 = np.sqrt(1.0 - y1 * y1)
            return np.vstack((y1, y2))
        raise NotImplementedError()


class DTLZ4(DTLZ):
    """The DTLZ4 multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 4 of :cite:`2002:smoo-tp`
    and follows the reference implementation in :cite:`2008:shark`.

    Implements the true Pareto front as defined in \
    `this page <https://bit.ly/3bcIa97>`_.
    """

    _a: float

    def __init__(
        self,
        n_dimensions: int = 2,
        n_objectives: int = 2,
        rng: np.random.Generator = None,
        a: float = 10.0,
    ) -> None:
        self._a = a
        super().__init__(n_dimensions, n_objectives, rng)

    @property
    def name(self) -> str:
        return "DTLZ4"

    def evaluate(self, xs: np.ndarray, count: bool = True) -> np.ndarray:
        self._pre_evaluate(xs, count=count)
        values = np.empty(shape=(len(xs), self._n_objectives))
        xm = xs ** self._a
        for i in range(len(xs)):
            tmp = xs[i, self._n_dimensions - self._k :] - 0.5
            g = np.sum(tmp * tmp)
            half_pi = 0.5 * math.pi
            values[i, :] = np.repeat(1.0 + g, self._n_objectives)
            for a in range(self._n_objectives):
                for b in range(self._n_objectives - a - 1):
                    values[i, a] *= math.cos(xm[i, b] * half_pi)
                if a > 0:
                    values[i, a] *= math.sin(
                        xm[i, self._n_objectives - a - 1] * half_pi
                    )
        return values

    def pareto_front(self, num=50) -> np.ndarray:
        if self._n_objectives == 2:
            y1 = np.linspace(0.0, 1.0, num, True)
            y2 = np.sqrt(1.0 - y1 * y1)
            return np.vstack((y1, y2))
        raise NotImplementedError()


class DTLZ5(DTLZ):
    """The DTLZ5 multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in :cite:`2008:shark`.
    """

    _has_known_pareto_front = False

    @property
    def name(self) -> str:
        return "DTLZ5"

    def evaluate(self, xs: np.ndarray, count: bool = True) -> np.ndarray:
        self._pre_evaluate(xs, count=count)
        values = np.empty(shape=(len(xs), self._n_objectives))
        for i in range(len(xs)):
            tmp = xs[i, self._n_dimensions - self._k :] - 0.5
            g = np.sum(tmp * tmp)
            values[i, :] = np.repeat(1.0 + g, self._n_objectives)
            theta = np.repeat(np.pi / (4.0 * (1 + g)), self._n_dimensions)
            theta[0] = xs[i, 0] * 0.5 * math.pi
            theta[1:] *= 1.0 + 2.0 * g * xs[i, 1:]
            for a in range(self._n_objectives):
                for b in range(self._n_objectives - a - 1):
                    values[i, a] *= math.cos(theta[b])
                if a > 0:
                    values[i, a] *= math.sin(theta[self._n_objectives - a - 1])
        return values


class DTLZ6(DTLZ):
    """The DTLZ6 multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in :cite:`2008:shark`.
    """

    _has_known_pareto_front = False

    @property
    def name(self) -> str:
        return "DTLZ6"

    def evaluate(self, xs: np.ndarray, count: bool = True) -> np.ndarray:
        self._pre_evaluate(xs, count=count)
        values = np.empty(shape=(len(xs), self._n_objectives))
        for i in range(len(xs)):
            g = np.sum(xs[i, self._n_dimensions - self._k :] ** 0.1)
            values[i, :] = np.repeat(1.0 + g, self._n_objectives)
            theta = np.repeat(np.pi / (4.0 * (1 + g)), self._n_dimensions)
            theta[0] = xs[i, 0] * 0.5 * math.pi
            theta[1:] *= 1.0 + 2.0 * g * xs[i, 1:]
            for a in range(self._n_objectives):
                for b in range(self._n_objectives - a - 1):
                    values[i, a] *= math.cos(theta[b])
                if a > 0:
                    values[i, a] *= math.sin(theta[self._n_objectives - a - 1])
        return values


class DTLZ7(DTLZ):
    """The DTLZ7 multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in :cite:`2008:shark` and the true \
    Pareto front as in `this page <https://bit.ly/3hUuMIk>`_.
    """

    _has_continuous_pareto_front = False

    @property
    def name(self) -> str:
        return "DTLZ7"

    def evaluate(self, xs: np.ndarray, count: bool = True) -> np.ndarray:
        self._pre_evaluate(xs, count=count)
        values = np.empty(shape=(len(xs), self._n_objectives))
        for i in range(len(xs)):
            g = (
                1.0
                + 9.0 * np.sum(xs[i, self._n_dimensions - self._k :]) / self._k
            )
            values[i, :] = xs[i, : self._n_objectives]
            h = self._n_objectives - np.sum(
                (values[i, :-1] / (1.0 + g))
                * (1.0 + np.sin(3.0 * np.pi * values[i, :-1]))
            )
            values[i, -1] = (1.0 + g) * h
        return values

    def pareto_front(self, num=50) -> np.ndarray:
        if self._n_objectives == 2:
            size = num // 4
            y1 = np.zeros(num)
            y1[:size] = np.linspace(0.0, 0.2514118360, size, True)
            y1[size : size * 2] = np.linspace(
                0.6316265307 + 1e-6, 0.8594008566, size, True
            )
            y1[size * 2 : size * 3] = np.linspace(
                1.3596178367 + 1e-6, 1.5148392681, size, True
            )
            y1[size * 3 :] = np.linspace(
                2.0518383519 + 1e-6, 2.116426807, size + num % 4, True
            )
            y2 = 4.0 - y1 * (1.0 + np.sin(3.0 * np.pi * y1))
            return np.vstack((y1, y2))
        raise NotImplementedError()


__all__ = [
    "DTLZ1",
    "DTLZ2",
    "DTLZ3",
    "DTLZ4",
    "DTLZ5",
    "DTLZ6",
    "DTLZ7",
]
