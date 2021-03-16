"""This module contains implementations of the CIGTAB family of benchmark \
functions."""
import numpy as np

from anguilla.fitness.base import ObjectiveFunction
from anguilla.util import random_orthogonal_matrix


class CIGTAB(ObjectiveFunction):
    """Common functionality for the CIGTAB family of functions."""

    def __init__(
        self,
        n_dimensions: int = 2,
        a: float = 1e3,
        rng: np.random.Generator = None,
        rotate: bool = True,
    ) -> None:
        self._a = a
        self._rotate = rotate
        super().__init__(n_dimensions, 2, rng)

    @property
    def name(self) -> str:
        return "CIGTAB"

    @property
    def qualified_name(self) -> str:
        return "{}(a={:.2E})".format(self.name, self._a)

    def evaluate(self, xs: np.ndarray, count: bool = True) -> np.ndarray:
        self._pre_evaluate(xs, count=count)
        n_points = len(xs)
        ys = np.empty_like(xs)
        zs = np.empty_like(xs)
        for i in range(n_points):
            ys[i] = self._rotation_matrix_y @ xs[i]
            zs[i] = self._rotation_matrix_z @ xs[i] - 2.0
        ys *= ys
        zs *= zs
        a_sqr = self._a * self._a
        values = np.empty((n_points, self._n_objectives))
        values[:, 0] = (
            ys[:, 0]
            + np.sum(self._a * ys[:, 1:-1], axis=1)
            + a_sqr * ys[:, -1]
        )
        values[:, 1] = (
            zs[:, 0]
            + np.sum(self._a * zs[:, 1:-1], axis=1)
            + a_sqr * zs[:, -1]
        )
        values /= self._scaler_inv
        return values


class CIGTAB1(CIGTAB):
    """The CIGTAB1 multi-objective benchmark function.

    Notes
    -----
    Implements the function as described in p. 15 of :cite:`2007:mo-cma-es`.
    """

    @property
    def name(self) -> str:
        return "CIGTAB1"

    def _post_update_n_dimensions(self) -> None:
        if self._rotate:
            self._rotation_matrix_y = random_orthogonal_matrix(
                self._n_dimensions, self._rng
            )
            self._rotation_matrix_z = self._rotation_matrix_y[:, :]
        else:
            self._rotation_matrix_y = np.eye(self._n_dimensions)
            self._rotation_matrix_z = self._rotation_matrix_y[:, :]

        self._scaler_inv = self._a * self._a * self._n_dimensions


class CIGTAB2(CIGTAB):
    """The CIGTAB2 multi-objective benchmark function.

    Notes
    -----
    Implements the function as described in p. 15 of :cite:`2007:mo-cma-es`.
    """

    @property
    def name(self) -> str:
        return "CIGTAB2"

    def _post_update_n_dimensions(self) -> None:
        if self._rotate:
            self._rotation_matrix_y = random_orthogonal_matrix(
                self._n_dimensions, self._rng
            )
            self._rotation_matrix_z = random_orthogonal_matrix(
                self._n_dimensions, self._rng
            )
        else:
            self._rotation_matrix_y = np.eye(self._n_dimensions)
            self._rotation_matrix_z = np.eye(self._n_dimensions)

        self._scaler_inv = self._a * self._a * self._n_dimensions


__all__ = [
    "CIGTAB1",
    "CIGTAB2",
]
