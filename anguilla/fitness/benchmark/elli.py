"""This module contains implementations of the ELLI family of benchmark \
functions."""
import numpy as np

from anguilla.fitness.base import AbstractObjectiveFunction
from anguilla.util import random_orthogonal_matrix


class ELLI1(AbstractObjectiveFunction):
    """The ELLI1 multi-objective unconstrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 15 of :cite:`2007:mo-cma-es`.
    """

    def __init__(
        self, n_dimensions: int, a: float, rng: np.random.Generator
    ) -> None:
        self._a = a
        super().__init__(n_dimensions, 2, rng)

    @property
    def name(self) -> str:
        return "ELLI1"

    def _post_update_n_dimensions(self) -> None:
        self._rotation_matrix = random_orthogonal_matrix(
            self._n_dimensions, self._rng
        )
        a_sq = self._a * self._a
        self._scaler = 1.0 / (a_sq * self._n_dimensions)
        self._exp = (a_sq) ** (
            np.arange(self._n_dimensions) / self._n_dimensions
        )

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_single_evaluation(x)
        y = self._rotation_matrix @ x
        value = np.empty(self._n_objectives)
        value[0] = self._scaler * np.sum(self._exp * y)
        tmp = (y - 2.0) * (y - 2.0)
        value[1] = self._scaler * np.sum(self._exp * tmp)
        return value

    def evaluate_multiple(self, xs: np.ndarray) -> np.ndarray:
        xs = self._pre_multiple_evaluation(xs)
        n_points = len(xs)
        y = np.empty_like(xs)
        for i in range(n_points):
            y[i] = self._rotation_matrix @ xs[i]
        values = np.empty((n_points, self._n_objectives))
        values[:, 0] = self._scaler * np.sum(self._exp * y, axis=1)
        tmp = (y - 2.0) * (y - 2.0)
        values[:, 1] = self._scaler * np.sum(self._exp * tmp, axis=1)
        return values
