"""This module contains an implementation of the ellipsoid benchmark function."""
from typing import Union, Optional

import numpy as np

from anguilla.fitness.base import ObjectiveFunction
from anguilla.util import random_orthogonal_matrix


class Ellipsoid(ObjectiveFunction):
    """The Ellipsoid single-objective function.

    Notes
    -----
    Implemented as described in :cite:`2007:mo-cma-es`.
    """

    _rotate: bool
    _rotation_matrix: Optional[np.ndarray]
    _alpha: float

    def __init__(
        self,
        n_dimensions: int = 2,
        alpha: float = 1e-3,
        rotate: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._alpha = alpha
        self._rotate = rotate
        super().__init__(n_dimensions, 1, rng)

    @property
    def name(self) -> str:
        return "Ellipsoid"

    @property
    def qualified_name(self) -> str:
        return "{}(alpha={:.2E})".format(self.name, self._a)

    def _post_update_n_dimensions(self) -> None:
        if self._rotate:
            self._rotation_matrix = random_orthogonal_matrix(
                self._n_dimensions, self._rng
            )
        n_dimensions_f = float(self.n_dimensions)
        self._scaler = self._alpha ** (
            np.arange(self._n_dimensions) / (n_dimensions_f - 1.0 + 1e-10)
        )

    def evaluate_single(self, x: np.ndarray) -> Union[float, np.ndarray]:
        self._pre_evaluate_single(x)
        y = self._rotation_matrix @ x if self._rotate else x
        return np.sum(self._scaler * (y * y), axis=0)

    def evaluate_multiple(self, xs: np.ndarray) -> Union[float, np.ndarray]:
        xs = self._pre_evaluate_multiple(xs)
        if self._rotate:
            ys = np.zeros_like(xs)
            for i in range(len(xs)):
                ys[i] = self._rotation_matrix @ xs[i]
        else:
            ys = xs
        values = np.sum(self._scaler * (ys * ys), axis=1)
        return values if len(xs) > 1 else values[0]
