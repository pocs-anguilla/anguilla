"""This module contains an implementation of the sphere benchmark function."""
from typing import Union

import numpy as np

from anguilla.fitness.base import AbstractObjectiveFunction


class Sphere(AbstractObjectiveFunction):
    """The Sphere single-objective function.

    Notes
    -----
    `Learn more <https://www.sfu.ca/~ssurjano/spheref.html>`_ about this \
    function at :cite:`2013:simulationlib`.
    """

    @property
    def name(self) -> str:
        return "Sphere"

    def evaluate_single(self, x: np.ndarray) -> Union[float, np.ndarray]:
        if len(x.shape) > 1:
            axis = 1
            self._evaluation_count += x.shape[0]
        else:
            axis = 0
            self._evaluation_count += 1
        return np.sum(x * x, axis=axis)

    evaluate_multiple = evaluate_single
