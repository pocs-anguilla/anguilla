"""This module contains an implementation of the Sphere benchmark function."""
import math

import numpy as np

from anguilla.fitness.base import ObjectiveFunction
from anguilla.fitness.constraints import BoxConstraintsHandler


class FON(ObjectiveFunction):
    """The FON multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 14 of :cite:`2007:mo-cma-es`.
    """

    _n_sqrt: float

    def __init__(
        self, n_dimensions: int = 2, rng: np.random.Generator = None
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

    def evaluate(self, xs: np.ndarray, count: bool = True) -> np.ndarray:
        self._pre_evaluate(xs, count=count)
        values = np.zeros((len(xs), self._n_objectives))
        tmp1 = xs - (1.0 / self._n_sqrt)
        tmp2 = xs + (1.0 / self._n_sqrt)
        values[:, 0] = 1.0 - np.exp(-np.sum(tmp1 * tmp1, axis=1))
        values[:, 1] = 1.0 - np.exp(-np.sum(tmp2 * tmp2, axis=1))
        return values
