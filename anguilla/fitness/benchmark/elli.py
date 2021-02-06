"""This module contains implementations of the ELLI family of benchmark \
functions."""
import math

import numpy as np

from anguilla.fitness.base import ObjectiveFunction
from anguilla.util import random_orthogonal_matrix


class ELLI(ObjectiveFunction):
    """Common functionality for the ELLI family of functions."""

    def __init__(
        self,
        n_dimensions: int = 2,
        a: float = 1e6,
        rng: np.random.Generator = None,
        rotate: bool = True,
    ) -> None:
        self._a = a
        self._rotate = rotate
        super().__init__(n_dimensions, 2, rng)

    @property
    def name(self) -> str:
        return "ELLI"

    @property
    def qualified_name(self) -> str:
        return "{}(a={:.2E})".format(self.name, self._a)

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_evaluate_single(x)
        y = self._rotation_matrix_y @ x
        y *= y
        z = self._rotation_matrix_z @ x - 2.0
        z *= z
        value = np.zeros(self._n_objectives)
        for i in range(self._n_dimensions):
            value[0] += self._coefficients[i] * y[i]
            value[1] += self._coefficients[i] * z[i]
        value /= self._scaler_inv
        return value

    def evaluate_multiple(self, xs: np.ndarray) -> np.ndarray:
        xs = self._pre_evaluate_multiple(xs)
        n_points = len(xs)
        ys = np.empty_like(xs)
        zs = np.empty_like(xs)
        for i in range(n_points):
            ys[i] = self._rotation_matrix_y @ xs[i]
            zs[i] = self._rotation_matrix_z @ xs[i] - 2.0
        ys *= ys
        zs *= zs
        values = np.empty((n_points, self._n_objectives))
        values[:, 0] = np.sum(self._coefficients * ys, axis=1)
        values[:, 1] = np.sum(self._coefficients * zs, axis=1)
        values /= self._scaler_inv
        return values if n_points > 1 else values[0]


class ELLI1(ELLI):
    """The ELLI1 multi-objective unconstrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 15 of :cite:`2007:mo-cma-es`
    and is also based on the implementation in :cite:`2008:shark`.
    """

    @property
    def name(self) -> str:
        return "ELLI1"

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
        self._coefficients = self._a ** (
            2.0 * np.arange(self._n_dimensions) / (self._n_dimensions - 1)
        )


class ELLI2(ELLI):
    """The ELLI2 multi-objective unconstrained benchmark function.

    Notes
    -----
    Implements the function as defined in p. 15 of :cite:`2007:mo-cma-es`.
    """

    @property
    def name(self) -> str:
        return "ELLI2"

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
        self._coefficients = self._a ** (
            2.0 * np.arange(self._n_dimensions) / (self._n_dimensions - 1)
        )


class GELLI(ObjectiveFunction):
    """
    The GELLI benchmark function.

    Notes
    -----
    Implements the function defined in p. 490 :cite:`2010:mo-cma-es`.
    """

    _has_scalable_dimensions = True
    _has_scalable_objectives = True

    def __init__(
        self,
        n_dimensions: int = 2,
        n_objectives: int = 2,
        a: float = 1000,
        d: float = 2,
        rng: np.random.Generator = None,
    ) -> None:
        self._a = a
        self._d = d
        self._centers_matrix = None
        super().__init__(n_dimensions, n_objectives, rng)

    @property
    def name(self) -> str:
        return "GELLI{}".format(self._n_objectives)

    @property
    def qualified_name(self) -> str:
        return "{}(a={:.2E}, d={:.2f})".format(self.name, self._a, self._d)

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_evaluate_single(x)
        value = np.zeros(self._n_objectives)
        v = self._rotation_matrix @ x
        for m in range(self._n_objectives):
            tmp = v - self._centers_matrix[m, :]
            value[m] = np.sum(tmp * tmp)
        return value

    def evaluate_multiple(self, xs: np.ndarray) -> np.ndarray:
        xs = self._pre_evaluate_multiple(xs)
        values = np.zeros((len(xs), self._n_objectives))
        n_points = len(xs)
        for i in range(n_points):
            v = self._rotation_matrix @ xs[i]
            for m in range(self._n_objectives):
                tmp = v - self._centers_matrix[m, :]
                values[i, m] = np.sum(tmp * tmp)
        return values if n_points > 1 else values[0]

    def _pre_update_n_dimensions(self, n_dimensions: int) -> None:
        # Test that m <= n
        if self._n_objectives > n_dimensions:
            raise ValueError(
                "n_objectives must be less than equal to n_dimensions"
            )

    def _post_update_n_dimensions(self) -> None:
        self._scaler_inv = self._a * self._a * self._n_dimensions
        # The O matrix in the paper
        orthogonal_matrix = random_orthogonal_matrix(
            self._n_dimensions, self._rng
        )
        # The D matrix in the paper
        diagonal_matrix = np.zeros((self._n_dimensions, self._n_dimensions))
        for i in range(self._n_dimensions):
            diagonal_matrix[i, i] = self._a ** (i / (self._n_dimensions - 1))
        # The DO matrix in the paper
        self._rotation_matrix = diagonal_matrix @ orthogonal_matrix
        # The M matrix in the paper
        self._create_centers_matrix()

    def _pre_update_n_objectives(self, n_objectives: int) -> None:
        # Test that m <= n
        if n_objectives > self._n_dimensions:
            raise ValueError(
                "n_objectives must be less than equal to n_dimensions"
            )

    def _post_update_n_objectives(self) -> None:
        # The M matrix in the paper
        self._create_centers_matrix()

    def _create_centers_matrix(self):
        self._centers_matrix = np.zeros(
            (self._n_objectives, self._n_dimensions)
        )
        d1 = math.sqrt((self._d - 1.0) / self._d)
        d2 = -1.0 / (math.sqrt(self._d * (self._d - 1.0)))
        for i in range(self._n_objectives):
            for j in range(self._n_dimensions):
                if i == j:
                    self._centers_matrix[i, j] = d1
                elif j <= self._n_objectives:
                    self._centers_matrix[i, j] = d2


__all__ = ["ELLI1", "ELLI2", "GELLI"]
