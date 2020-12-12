"""This module contains an implementation of the Sphere benchmark function."""
import math

import numpy as np

from anguilla.fitness.base import AbstractObjectiveFunction
from anguilla.fitness.constraints import BoxConstraintsHandler


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
        super().__init__(n_dimensions, 2, rng)

    @property
    def name(self) -> str:
        return "FON"

    def _post_update_n_dimensions(self) -> None:
        self._constraints_handler = BoxConstraintsHandler(
            self._n_dimensions, (-4.0, 4.0)
        )
        self._n_sqrt = math.sqrt(self._n_dimensions)

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_single_evaluation(x)
        value = np.zeros(self._n_objectives)
        tmp1 = x - self._n_sqrt
        tmp2 = x + self._n_sqrt
        value[0] = 1.0 - math.exp(-np.sum(tmp1 * tmp1))
        value[1] = 1.0 - math.exp(-np.sum(tmp2 * tmp2))
        return value

    def evaluate_multiple(self, xs: np.ndarray) -> np.ndarray:
        xs = self._pre_multiple_evaluation(xs)
        values = np.zeros((len(xs), self._n_objectives))
        tmp1 = xs - self._n_sqrt
        tmp2 = xs + self._n_sqrt
        values[:, 0] = 1.0 - np.exp(-np.sum(tmp1 * tmp1, axis=1))
        values[:, 1] = 1.0 - np.exp(-np.sum(tmp2 * tmp2, axis=1))
        return values
