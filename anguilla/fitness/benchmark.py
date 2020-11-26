"""This module contains benchmark fitness/objective functions."""

import numba
import numpy as np

from anguilla.fitness.base import AbstractObjectiveFunctionWithInputDomain

# We use Numba to implement fitness functions that support
# Numpy's "Generalized Universal Function API".
# Ref.: https://numpy.org/doc/stable/reference/c-api/generalized-ufuncs.html


@numba.guvectorize([(numba.f8[:], numba.f8[:])], "(n)->()")
def _sphere(x, y):
    y[0] = np.sum(x ** 2)


@numba.guvectorize([(numba.f8[:], numba.f8[:])], "(n)->()")
def _sum_squares(x, y):
    ii = np.arange(1, x.shape[0] + 1, dtype=numba.f8)
    y[0] = np.sum((x ** 2) * ii)


class Sphere(AbstractObjectiveFunctionWithInputDomain):
    """The Sphere function.

    Notes
    -----

    `Learn more <https://www.sfu.ca/~ssurjano/spheref.html>`_ about this function at :cite:`2013:simulationlib`.
    """

    x_min = -5.12
    x_max = 5.12

    def __init__(self, rng=None):
        super().__init__(_sphere, rng=rng)

    def propose_initial_point(self, n: int, m: int = 1) -> np.ndarray:
        size = (m, n) if m > 1 else n
        return self._rng.standard_normal(size)


class SumSquares(AbstractObjectiveFunctionWithInputDomain):
    """The Axis Parallel Hyper-Ellipsoid function.

    Notes
    -----

    `Learn more <https://www.sfu.ca/~ssurjano/sumsqu.html>`_ about this function at :cite:`2013:simulationlib`.
    """

    x_min = -10.0
    x_max = 10.0

    def __init__(self, rng=None):
        super().__init__(_sum_squares, rng=rng)

    def propose_initial_point(self, n: int, m: int = 1) -> np.ndarray:
        size = (m, n) if m > 1 else n
        return self._rng.standard_normal(size)
