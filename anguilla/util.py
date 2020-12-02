"""This module contains utilitary/auxilliary functions."""
import math
import functools

from typing import Callable

import numba
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
    """Generate a random orthogonal matrix with distribution given by a \
    Haar measure.

    Parameters
    ----------
    n
        Size of the matrix.
    rng
        A random number generator.

    Returns
    -------
    np.ndarray
        The random orthogonal matrix.

    Notes
    -----
    Implements the algorithm used in :cite:`2008:shark`, where it was chosen \
    because it is faster [#f1]_ than an alternative algorithm that uses the \
    QR decomposition and corrects the signs. See p. 16 from \
    :cite:`2007:random-matrices`. We port the implementation here.

    Some benchmark functions (e.g. CIGTAB1) require an orthogonal rotation \
    matrix :cite:`2007:mo-cma-es`, so it suffices to use this variant over \
    others that produce unitary random matrices.

    .. [#f1] In our experiments, approximately twice as fast.
    """
    v = rng.standard_normal((n * n + n - 2) // 2)
    return _random_orthogonal_matrix_jit(n, v)


@numba.njit
def _random_orthogonal_matrix_jit(n: int, v: np.ndarray) -> np.ndarray:
    """The optimized algorithm implementation for \
    :py:mod:`random_orthogonal_matrix`.
    
    Parameters
    ----------
    n
        Size of the matrix.
    v
        Vector with (n^2 + n - 2) / 2 realizations of the starndard normal \
        distribution.

    Returns
    -------
    np.ndarray
        The random orthogonal matrix.
    """
    Q = np.eye(n)
    k = 0
    # We can safely skip the first iteration
    for i in range(2, n + 1):
        # Compute v_hat
        ki = k + i
        ni = n - i
        v[k:ki] /= np.linalg.norm(v[k:ki])
        # Compute u_hat
        sgn = v[k] / abs(v[k])
        v[k] += sgn
        tmp = v[k:ki]
        beta = 2.0 / np.sum(tmp * tmp)
        # Apply the Householder reflection
        Q[ni:n, ni:n] -= beta * np.outer(tmp, Q[ni:n, ni:n].T @ tmp)
        Q[ni:n, ni:n] *= -sgn
        k += i
    return Q
