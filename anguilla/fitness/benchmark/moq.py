"""This module contains implementations of the family of multi-objective \
quadratic benchmark functions presented in \
:cite:`2019:mo-quadratic-benchmark`."""
import re
import math
from typing import Optional

import numpy as np

from anguilla.fitness.base import ObjectiveFunction, BoundsTuple
from anguilla.fitness.constraints import BoxConstraintsHandler
from anguilla.util import random_orthogonal_matrix


PROBLEM_NAME_FORMAT = re.compile(r"^[1-9][/|][CIJ]$")


class MOQ(ObjectiveFunction):
    """Generate a problem from multi-objective quadratic benchmark.

    Parameters
    ----------
    name
        The problem class name.
    n_dimensions
        The dimensionality of the search space.
    k
        The conditioning number.
    rng
        A random number generator.

    Raises
    ------
    ValueError
        Invalid value for a parameter.

    Notes
    -----
    The value for `name` should a string with length 3, with this format:

    * The first character is the problem case (valid values: `[1-9]`).
    * The second character is the alignment of the generalized eigenvector \
      (valid values: `|` for aligned or `/` for non-aligned).
    * The third character is the shape of the Pareto front (valid values: \
      `I` for linear, `C` for convex or `J` for concave)

    Implements the convex quadratic benchmark functions defined in \
    [2019:mo-quadratic-benchmark] and is based on the code from the \
    supplementary material of the paper.
    """

    _has_known_pareto_front = True

    def __init__(
        self,
        name: str,
        n_dimensions: int = 2,
        k: float = 1e3,
        rng: np.random.Generator = None,
    ) -> None:
        if PROBLEM_NAME_FORMAT.match(name) is None:
            raise ValueError("Invalid value for name")
        self._name = name
        self._category = int(name[0])
        self._aligned = name[1] == "|"
        if name[2] == "C":
            # Convex
            self._shape = 1.0
        elif name[2] == "I":
            # Linear
            self._shape = 0.5
        else:
            # Concave
            self._shape = 0.25
        self._k = k
        # Note: The definition allows for scalable number of objectives.
        super().__init__(n_dimensions, 2, rng)

    @property
    def name(self) -> str:
        return self._name

    @property
    def qualified_name(self) -> str:
        return "{}(k={:.2E})".format(self._name, self._k)

    def evaluate_single(self, x: np.ndarray) -> np.ndarray:
        self._pre_evaluate_single(x)
        value = np.zeros(self._n_objectives)
        for i in range(self._n_objectives):
            d = self._A[i].T @ (x - self._x[i])
            value[i] = (
                self._a[i] * 0.5 * np.power(np.inner(d, d), self._shape)
                + self._b[i]
            )
        return value

    def evaluate_multiple(self, xs: np.ndarray) -> np.ndarray:
        xs = self._pre_evaluate_multiple(xs)
        values = np.zeros((len(xs), self._n_objectives))
        for j in range(len(xs)):
            for i in range(self._n_objectives):
                d = self._A[i].T @ (xs[j] - self._x[i])
                values[j, i] = (
                    self._a[i] * 0.5 * np.power(np.inner(d, d), self._shape)
                    + self._b[i]
                )
        return values if len(xs) > 1 else values[0]

    def pareto_front(self, num: int = 50) -> np.ndarray:
        # [2019:mo-quadratic-benchmark] p.4
        ts = np.linspace(0.0, 1.0, endpoint=True, num=num)
        # Compute components of the Pareto front
        y1 = np.zeros(num)
        y2 = np.zeros(num)
        for i in range(num):
            x = self._x[0] * (1.0 - ts[i]) + self._x[1] * ts[i]
            y1[i], y2[i] = self.evaluate_single(x)
        return np.stack((y1, y2))

    def random_points(
        self,
        m: int = 1,
        region_bounds: Optional[BoundsTuple] = None,
    ) -> np.ndarray:
        return np.zeros((m, self._n_dimensions))

    def _post_update_n_dimensions(self) -> None:
        n_dimensions = self._n_dimensions
        n_objectives = self._n_objectives
        self._delta = np.zeros(n_dimensions)
        self._H = np.zeros((n_objectives, n_dimensions, n_dimensions))
        self._A = np.zeros((n_objectives, n_dimensions, n_dimensions))
        self._U = np.zeros((n_objectives, n_dimensions, n_dimensions))
        self._d = np.ones((n_objectives, n_dimensions))
        self._a = np.zeros(n_objectives)
        self._b = np.zeros(n_objectives)
        self._x = np.zeros((n_objectives, n_dimensions))
        delta_from_gev = False
        # The next function call produces side-effects in _U, _d, _delta
        if self._category == 1:
            delta_from_gev = self._post_update_n_dimensions_1()
        elif self._category == 2:
            delta_from_gev = self._post_update_n_dimensions_2()
        elif self._category == 3:
            delta_from_gev = self._post_update_n_dimensions_3()
        elif self._category == 4:
            delta_from_gev = self._post_update_n_dimensions_4()
        elif self._category == 5:
            delta_from_gev = self._post_update_n_dimensions_5()
        elif self._category == 6:
            delta_from_gev = self._post_update_n_dimensions_6()
        elif self._category == 7:
            delta_from_gev = self._post_update_n_dimensions_7()
        elif self._category == 8:
            delta_from_gev = self._post_update_n_dimensions_8()
        else:
            delta_from_gev = self._post_update_n_dimensions_9()
        # Compute the Cholesky factors from the eigen decomposition matrices
        # and the Hessian from the Cholesky factors
        for i in range(n_objectives):
            self._A[i] = self._U[i] @ np.diag(np.sqrt(self._d[i]))
            self._H[i] = self._A[i] @ self._A[i].T
        # Compute delta if pending
        if delta_from_gev:
            A2_inv = (
                self._U[1] @ np.diag(1.0 / np.sqrt(self._d[1])) @ self._U[1].T
            )
            _, V = np.linalg.eigh(A2_inv @ self._H[0] @ A2_inv)
            self._delta = V[:, self._rng.integers(self._n_dimensions)]
            self._delta /= np.linalg.norm(self._delta, ord=2)
        # Set optimal solution
        center = self._rng.standard_normal(size=self._n_dimensions)
        while np.linalg.norm(center, ord=np.inf) >= 4.5:
            center = self._rng.standard_normal(size=self._n_dimensions)
        self._x[0] = center - 0.5 * self._delta
        self._x[1] = center + 0.5 * self._delta
        for i in range(self._n_objectives):
            self._a[i] = 10.0 ** (6.0 * self._rng.uniform())
            self._b[i] = 2.0 * self._a[i] * self._rng.uniform() - self._a[i]
        self._constraints_handler = BoxConstraintsHandler(
            self._n_dimensions, (-5.0, 5.0)
        )

    def _post_update_n_dimensions_1(self) -> bool:
        # U1 = U2 = I, D1 = D2 = I
        for i in range(self._n_dimensions):
            self._U[:, i, i] = 1.0
        if self._aligned:
            self._delta[self._rng.integers(self._n_dimensions)] = 1.0
        else:
            self._delta = self._rng.standard_normal(size=self._n_dimensions)
            self._delta /= np.linalg.norm(self._delta, ord=2)
        return False

    def _post_update_n_dimensions_2(self) -> bool:
        # U1 = U2 = I, D1 = I != D2
        for i in range(self._n_dimensions):
            self._U[:, i, i] = 1.0
        if self._aligned:
            self._d[1] = self._random_d()
            self._delta[self._rng.integers(self._n_dimensions)] = 1.0
        else:
            i = self._rng.integers(self._n_dimensions)
            j = self._rng.integers(self._n_dimensions - 2)
            if j >= i:
                j += 1
            self._d[1] = self._random_d_dup(i, j)
            angle = 2.0 * math.pi * self._rng.uniform(1.0)
            self._delta[i] = math.cos(angle)
            self._delta[j] = math.sin(angle)
        return False

    def _post_update_n_dimensions_3(self) -> None:
        # U1 = U2 = I
        for i in range(self._n_dimensions):
            self._U[:, i, i] = 1.0
        raise NotImplementedError()

    def _post_update_n_dimensions_4(self) -> None:
        # U1 = U2 = I
        for i in range(self._n_dimensions):
            self._U[:, i, i] = 1.0
        raise NotImplementedError()

    def _post_update_n_dimensions_5(self) -> None:
        # U1 = I, U2 != I, D2 != I
        for i in range(self._n_dimensions):
            self._U[0, i, i] = 1.0
        self._U[1] = random_orthogonal_matrix(self._n_dimensions, self._rng)
        self._d[1] = self._random_d()
        raise NotImplementedError()

    def _post_update_n_dimensions_6(self) -> None:
        # U1 = I, U2 != I, D2 != I
        for i in range(self._n_dimensions):
            self._U[0, i, i] = 1.0
        self._U[1] = random_orthogonal_matrix(self._n_dimensions, self._rng)
        self._d[1] = self._random_d()
        raise NotImplementedError()

    def _post_update_n_dimensions_7(self) -> None:
        # U1 = U2 != I, D1 != I != D2
        self._U[0] = random_orthogonal_matrix(self.n_dimensions, self._rng)
        self._U[1] = self._U[0]
        self._d[0] = self._random_d()
        self._d[1] = self._random_d()
        raise NotImplementedError()

    def _post_update_n_dimensions_8(self) -> None:
        # U1 = U2 != I, D1 != I != D2
        self._U[0] = random_orthogonal_matrix(self.n_dimensions, self._rng)
        self._U[1] = self._U[0]
        self._d[0] = self._random_d()
        self._d[1] = self._random_d()
        raise NotImplementedError()

    def _post_update_n_dimensions_9(self) -> None:
        # I != U1 != U2 != I, D1 != I != D2
        self._U[0] = random_orthogonal_matrix(self._n_dimensions, self._rng)
        self._U[1] = random_orthogonal_matrix(self._n_dimensions, self._rng)
        self._d[0] = self._random_d()
        self._d[1] = self._random_d()
        raise NotImplementedError()

    def _random_d(self) -> np.ndarray:
        """Return the shuffled eigenvalue spectrum of an ellipsoid function."""
        n_dimensions = self._n_dimensions
        d = self._k ** (np.arange(n_dimensions) / (n_dimensions - 1))
        self._rng.shuffle(d)
        return d

    def _random_d_dup(self, i: int, j: int) -> np.ndarray:
        """Return the shuffled eigenvalue spectrum of (n-1) dimensional\
        ellipsoid function with a duplicate eigenvalue at random."""
        if j < i:
            i, j = j, i
        k = self._n_dimensions - 1
        d = self._k ** (np.arange(k + 1) / (k - 1))
        self._rng.shuffle(d[0:k])
        d[k] = d[i]
        if j != k:
            d[k], d[j] = d[j], d[k]
        return d
