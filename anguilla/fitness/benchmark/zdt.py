"""This module contains implementations of the ZDT family of benchmark \
functions."""
import numpy as np

from anguilla.fitness.base import AbstractObjectiveFunction
from anguilla.fitness.constraints import BoxConstraintsHandler
from anguilla.util import random_orthogonal_matrix


class ZDT4P(AbstractObjectiveFunction):
    """The ZDT4' multi-objective box-constrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 16 of :cite:`2007:mo-cma-es`.
    """

    def __init__(
        self, n_dimensions: int, rng: np.random.Generator = None
    ) -> None:
        super().__init__(n_dimensions, 2, rng)

    @property
    def name(self) -> str:
        return "ZDT4'"

    def _post_update_n_dimensions(self) -> None:
        n = self._n_dimensions
        # Update box constraints
        lower_bounds = np.repeat(-5.0, n)
        lower_bounds[0] = 0.0
        upper_bounds = np.repeat(5.0, n)
        upper_bounds[0] = 1.0
        self._constraints_handler = BoxConstraintsHandler(
            self._n_dimensions, (lower_bounds, upper_bounds)
        )
        # Create the random orthogonal (and restricted) rotation matrix
        self._rotation_matrix = np.zeros((n, n))
        self._rotation_matrix[0, 0] = 1.0
        self._rotation_matrix[1:, 1:] = random_orthogonal_matrix(
            n - 1, self._rng
        )

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_single_evaluation(x)
        # First objective
        value = np.array([x[0], 0.0])
        # Second objective
        y = self._rotation_matrix @ x

        value[1] = (
            1.0
            + (10.0 * float(self._n_dimensions - 1))
            + np.sum(y[1:] * y[1:] - 10.0 * np.cos(4.0 * np.pi * y[1:]))
        )

        value[1] *= 1.0 - np.sqrt(x[0] / value[1])
        return value

    def evaluate_multiple(self, xs: np.ndarray) -> np.ndarray:
        # Setup
        xs = self._pre_multiple_evaluation(xs)
        n_points = len(xs)
        shape = (n_points, self._n_objectives)
        values = np.zeros(shape)
        # First objective
        values[:, 0] = xs[:, 0]
        # Second objective
        n = self._n_dimensions
        y = np.zeros_like(xs)
        for i in range(n_points):
            y[i] = self._rotation_matrix @ xs[i]

        values[:, 1] = (
            1.0
            + (10.0 * float(n - 1))
            + np.sum(
                y[:, 1:] * y[:, 1:] - 10.0 * np.cos(4.0 * np.pi * y[:, 1:]),
                axis=1,
            )
        )

        values[:, 1] *= 1.0 - np.sqrt(values[:, 0] / values[:, 1])
        return values if len(xs.shape) > 1 else values[0]
