"""This module contains implementations of the DTLZ family of benchmark \
functions."""
import math

import numpy as np

from anguilla.fitness.base import ObjectiveFunction
from anguilla.fitness.constraints import BoxConstraintsHandler


class DTLZ(ObjectiveFunction):
    """Common functionality for the DTLZ family of functions."""

    def __init__(
        self,
        n_dimensions: int = 2,
        n_objectives: int = 2,
        rng: np.random.Generator = None,
    ) -> None:
        self._scalable_objectives = True
        super().__init__(n_dimensions, n_objectives, rng)

    @property
    def name(self) -> str:
        return "DTLZ"

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
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

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_evaluate_single(x)
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
        xs = self._pre_evaluate_multiple(xs)
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

    def pareto_front(self, num=50) -> np.ndarray:
        if self._n_objectives == 2:
            x1 = np.linspace(0.0, 0.5, num, True)
            x2 = 0.5 - x1
            return np.vstack((x1, x2))
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

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_evaluate_single(x)
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

    def pareto_front(self, num=50) -> np.ndarray:
        if self._n_objectives == 2:
            x1 = np.linspace(0.0, 1.0, num, True)
            x2 = np.sqrt(1.0 - x1)
            return np.vstack((x1, x2))
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

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_evaluate_single(x)
        tmp = x[self._n_dimensions - self._k :] - 0.5
        g = 100.0 * (self._k + np.sum(tmp * tmp - np.cos(20.0 * np.pi * tmp)))
        half_pi = 0.5 * math.pi
        value = np.repeat(1.0 + g, self._n_objectives)
        for i in range(self._n_objectives):
            for j in range(self._n_objectives - i - 1):
                value[i] *= math.cos(x[j] * half_pi)
            if i > 0:
                value[i] *= math.sin(x[self._n_objectives - i - 1] * half_pi)
        return value

    def pareto_front(self, num=50) -> np.ndarray:
        if self._n_objectives == 2:
            x1 = np.linspace(0.0, 1.0, num, True)
            x2 = np.sqrt(1.0 - x1)
            return np.vstack((x1, x2))
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

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_evaluate_single(x)
        tmp = x[self._n_dimensions - self._k :] - 0.5
        g = np.sum(tmp * tmp)
        half_pi = 0.5 * math.pi
        xm = x ** self._a
        value = np.repeat(1.0 + g, self._n_objectives)
        for i in range(self._n_objectives):
            for j in range(self._n_objectives - i - 1):
                value[i] *= math.cos(xm[j] * half_pi)
            if i > 0:
                value[i] *= math.sin(xm[self._n_objectives - i - 1] * half_pi)
        return value

    def pareto_front(self, num=50) -> np.ndarray:
        if self._n_objectives == 2:
            x1 = np.linspace(0.0, 1.0, num, True)
            x2 = np.sqrt(1.0 - x1)
            return np.vstack((x1, x2))
        raise NotImplementedError()


class DTLZ5(DTLZ):
    """The DTLZ5 multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in :cite:`2008:shark`.
    """

    @property
    def name(self) -> str:
        return "DTLZ5"

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_evaluate_single(x)
        tmp = x[self._n_dimensions - self._k :] - 0.5
        g = np.sum(tmp * tmp)
        value = np.repeat(1.0 + g, self._n_objectives)
        theta = np.repeat(np.pi / (4.0 * (1 + g)), self._n_dimensions)
        theta[0] = x[0] * 0.5 * math.pi
        theta[1:] *= 1.0 + 2.0 * g * x[1:]
        for i in range(self._n_objectives):
            for j in range(self._n_objectives - i - 1):
                value[i] *= math.cos(theta[j])
            if i > 0:
                value[i] *= math.sin(theta[self._n_objectives - i - 1])
        return value


class DTLZ6(DTLZ):
    """The DTLZ6 multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in :cite:`2008:shark`.
    """

    @property
    def name(self) -> str:
        return "DTLZ6"

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_evaluate_single(x)
        g = np.sum(x[self._n_dimensions - self._k :] ** 0.1)
        value = np.repeat(1.0 + g, self._n_objectives)
        theta = np.repeat(np.pi / (4.0 * (1 + g)), self._n_dimensions)
        theta[0] = x[0] * 0.5 * math.pi
        theta[1:] *= 1.0 + 2.0 * g * x[1:]
        for i in range(self._n_objectives):
            for j in range(self._n_objectives - i - 1):
                value[i] *= math.cos(theta[j])
            if i > 0:
                value[i] *= math.sin(theta[self._n_objectives - i - 1])
        return value


class DTLZ7(DTLZ):
    """The DTLZ7 multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in :cite:`2008:shark` and the true \
    Pareto front as in `this page <https://bit.ly/3hUuMIk>`_.
    """

    @property
    def name(self) -> str:
        return "DTLZ7"

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_evaluate_single(x)
        g = 1.0 + 9.0 * np.sum(x[self._n_dimensions - self._k :]) / self._k
        value = np.copy(x[: self._n_objectives])
        h = self._n_objectives - np.sum(
            (value[:-1] / (1.0 + g)) * (1.0 + np.sin(3.0 * np.pi * value[:-1]))
        )
        value[-1] = (1.0 + g) * h
        return value

    def pareto_front(self, num=50) -> np.ndarray:
        if self._n_objectives == 2:
            size = num // 4
            x1 = np.zeros(num)
            x1[:size] = np.linspace(0.0, 0.2514118360, size, True)
            x1[size : size * 2] = np.linspace(
                0.6316265307 + 1e-6, 0.8594008566, size, True
            )
            x1[size * 2 : size * 3] = np.linspace(
                1.3596178367 + 1e-6, 1.5148392681, size, True
            )
            x1[size * 3 :] = np.linspace(
                2.0518383519 + 1e-6, 2.116426807, size + num % 4, True
            )
            x2 = 4.0 - x1 * (1.0 + np.sin(3.0 * np.pi * x1))
            return np.vstack((x1, x2))
        raise NotImplementedError()
