"""This module contains utilitary/auxilliary functions."""
import math
import functools

from typing import Callable

import numpy as np


def _exp_norm_chi_implementation(
    f: Callable[[float], float]
) -> Callable[[float], float]:
    """Decorate exp_norm_chi so that it is possible to vary its \
        implementation while keeping a single copy of the docstring."""
    try:
        import scipy.special

        @functools.wraps(f)
        def implementation(k: float) -> float:
            tmp = k * 0.5
            return (
                math.sqrt(2.0)
                * scipy.special.gamma(tmp + 0.5)
                / scipy.special.gamma(tmp)
            )

    except ImportError:

        @functools.wraps(f)
        def implementation(k: float) -> float:
            return math.sqrt(2.0) * (
                1.0 - 1.0 / (4.0 * k) + 1.0 / (21.0 * k * k)
            )

    return implementation


@_exp_norm_chi_implementation
def exp_norm_chi(k: float) -> float:
    """Approximate the expectation of a random variable defined as \
        the 2-norm of another sampled from a k-dimensional \
            multivariate Gaussian distribution.

    Parameters
    ----------
    k
        The dimensionality of the multivariate Gaussian distribution.

    Returns
    -------
        An approximation of said expectation.

    Notes
    -----
    The formula is presented in p. 28 of :cite:`2016:cma-es-tutorial`:

    .. math::
        \\mathbb{E} [ ||Z|| ] = \\sqrt{2} \
        \\Gamma \\left( \\frac{k+1}{2} \\right) \
        / \\Gamma \\left( \\frac{k}{2} \\right) \\approx \\sqrt{k} \
            \\left(1 + \\frac{k+1}{2} + \\frac{1}{21 k^2} \\right)

    where

    .. math::
        Z \\sim N(0, I_k)

    If Scipy is available, use its gamma function implementation to \
        compute the approximation.
    """
    pass


def random_orthogonal_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random orthogonal matrix.

    Parameters
    ----------
    n
        Size of the matrix.
    rng
        A random number generator.

    Returns
    -------
    np.ndarray
        The random orthogonal matrix with distribution given by a Haar \
        measure.

    Notes
    -----
    Implements the sign correction of the Q matrix (obtained from QR \
    factorization of a random matrix sampled from the standard \
    normal distribution), so that the resulting distribution is a Haar \
    measure. The algorithm is described in :cite:`2007:random-matrices`.

    Our implementation of the alternative used in :cite:`2008:shark` does \
    not perform better (as measured by wall-clock time). Future work could be \
    to fix the implementation from the `prototype notebook \
    <https://git.io/JItU4>`_ to make it perform faster.
    """
    Z = rng.standard_normal(size=(n, n))
    Q, r = np.linalg.qr(Z)
    d = np.diag(r)
    return Q * (d / np.abs(d)) @ Q


def random_cliff_3d(n, rng=np.random.default_rng()):
    """Generate a random Cliff 3D front.

    Parameters
    ----------
    n
        Size of the front.
    rng
        A random number generator.

    Returns
    -------
    np.ndarray
        The random front.

    Notes
    -----
    Implements the Cliff3D as described on sec. 5 of \
    :cite:`2011:hypervolume-3d`.
    """
    vs = rng.standard_normal(size=(n, 2))
    ys = np.zeros((n, 3))
    for i in range(n):
        c = np.linalg.norm(vs[i])
        ys[i, 0:2] = 10.0 * np.abs(vs[i]) / c
    ys[:, 2] = rng.uniform(low=0.0, high=10.0, size=n)
    return ys


def random_3d_front(n, rng=np.random.default_rng(), dominated=False):
    """Generate a random 3D front.

    Parameters
    ----------
    n
        Size of the front.
    rng
        A random number generator.
    dominated
        If true, generates some dominated points.

    Returns
    -------
    np.ndarray
        The random 3D front and a reference point.
    """
    cliff3d = random_cliff_3d(n, rng=rng)
    dom = None
    if dominated:
        m = max(2, n // 3)
        dom = cliff3d[rng.choice(n, m, replace=False), :]
        for i in range(m):
            dom[i, rng.integers(0, 2)] += 0.01
    front = np.vstack([x for x in [cliff3d, dom] if x is not None])
    rng.shuffle(front)
    nadir = np.ceil(np.max(front, axis=0))
    return front, nadir


def random_2d_3d_front(n, dominated=False):
    """Generate a random pair of 2D and 3D fronts.

    Parameters
    ----------
    n
        Size of the fronts.
    rng
        A random number generator.
    dominated
        If true, generates some dominated points.

    Returns
    -------
    np.ndarray
        The random fronts and reference points.
    """
    front_3d, nadir_3d = random_3d_front(n, dominated=dominated)
    n = len(front_3d)
    front_3d[:, 2] = 0.0
    nadir_3d[2] = 1.0
    front_2d = np.delete(front_3d, 2, 1)
    nadir_2d = np.empty(2)
    nadir_2d[0] = nadir_3d[0]
    nadir_2d[1] = nadir_3d[1]
    return front_3d, nadir_3d, front_2d, nadir_2d


# Ported from Shark.
def random_kd_front(n: int, d: int, p: float, rng=np.random.default_rng()):
    """Utility function to create a random front.

    Parameters
    ----------
    n
        Size of the front.
    d
        Dimensionality (number of objectives).
    p
        Exponent. Determines whether the front is linear (p=1.0), \
        convex (p=2.0) or concave (p=0.5).
    rng
        A random number generator.

    Returns
    -------
    np.ndarray
        The random fronts and reference points.

    Notes
    -----
    Assumes that the reference point is a d-dimensional 'ones vector', \
    ie., [1,1..1].
    Ported from :cite:`2008:shark`, URL: https://git.io/Jtaci
    """
    points = np.zeros((n, d))
    for i in range(n):
        norm = 0.0
        sum = 0.0
        for j in range(d):
            points[i, j] = 1.0 - rng.uniform(0.0, 1.0 - sum)
            sum += 1.0 - points[i, j]
            norm += points[i, j] ** p
        norm = norm ** (1.0 / p)
        points[i] /= norm
    return points


def random_linear_front(n: int, d: int, rng=np.random.default_rng()):
    """Create a random linear front."""
    return random_kd_front(n, d, 1.0, rng)


def random_concave_front(n: int, d: int, rng=np.random.default_rng()):
    """Create a random linear front."""
    return random_kd_front(n, d, 0.5, rng)


def random_convex_front(n: int, d: int, rng=np.random.default_rng()):
    """Create a random linear front."""
    return random_kd_front(n, d, 2.0, rng)


__all__ = [
    "exp_norm_chi",
    "random_orthogonal_matrix",
    "random_cliff_3d",
    "random_3d_front",
    "random_2d_3d_front",
    "random_kd_front",
    "random_linear_front",
    "random_concave_front",
    "random_convex_front",
]
