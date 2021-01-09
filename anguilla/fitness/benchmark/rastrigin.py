"""This module contains an implementation of the Rastrigin benchmark function."""
from typing import Union, Optional

import numpy as np

from anguilla.fitness.base import ObjectiveFunction
from anguilla.util import random_orthogonal_matrix


class Rastrigin(ObjectiveFunction):
    """The Rastrigin single-objective function.

    Notes
    -----
    Implemented as described in :cite:`2007:mo-cma-es` and also follows
    the implementation in :cite:`2008:shark`.
    """

    _rotate: bool
    _rotation_matrix: Optional[np.ndarray]

    def __init__(
        self,
        n_dimensions: int = 2,
        rotate: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._rotate = rotate
        super().__init__(n_dimensions, 1, rng)

    @property
    def name(self) -> str:
        return "Rastrigin"

    def _post_update_n_dimensions(self) -> None:
        if self._rotate:
            self._rotation_matrix = random_orthogonal_matrix(
                self._n_dimensions, self._rng
            )

    def evaluate_single(self, x: np.ndarray) -> Union[float, np.ndarray]:
        self._pre_evaluate_single(x)
        y = self._rotation_matrix @ x if self._rotate else x
        # Use the same trick as Shark for numerical stability
        tmp = np.sin(np.pi * y)
        return np.sum(y * y) + 20.0 * np.sum(tmp * tmp)

    def evaluate_multiple(self, xs: np.ndarray) -> Union[float, np.ndarray]:
        xs = self._pre_evaluate_multiple(xs)
        if self._rotate:
            size = len(xs)
            ys = np.zeros_like(xs)
            for i in range(size):
                ys[i] = self._rotation_matrix @ xs[i]
        else:
            ys = xs
        tmp = np.sin(np.pi * ys)
        values = np.sum(ys * ys, axis=1) + 20.0 * np.sum(tmp * tmp, axis=1)
        return values if len(xs) > 1 else values[0]
