"""This module contains an implementation of the sphere benchmark function."""
from typing import Union, Optional

import numpy as np

from anguilla.fitness.base import ObjectiveFunction


class Sphere(ObjectiveFunction):
    """The Sphere single-objective function.

    Notes
    -----
    `Learn more <https://www.sfu.ca/~ssurjano/spheref.html>`_ about this \
    function at :cite:`2013:simulationlib`.
    """

    def __init__(
        self, n_dimensions: int = 2, rng: Optional[np.random.Generator] = None
    ) -> None:
        super().__init__(n_dimensions, 1, rng)

    @property
    def name(self) -> str:
        return "Sphere"

    def evaluate_single(self, x: np.ndarray) -> Union[float, np.ndarray]:
        self._pre_evaluate_single(x)
        return np.sum(x * x, axis=0)

    def evaluate_multiple(self, xs: np.ndarray) -> Union[float, np.ndarray]:
        xs = self._pre_evaluate_multiple(xs)
        values = np.sum(xs * xs, axis=1)
        return values if len(xs) > 1 else values[0]
