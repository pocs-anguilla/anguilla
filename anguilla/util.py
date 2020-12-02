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
    Implements the sign correction from :cite:`2007:random-matrices` of \
    the QR factorization of a random matrix sampled from the standard \
    normal distribution, so that the resulting distribution is a Haar \
    measure.

    Our implementation of the alternative used in :cite:`2008:shark` does \
    not perform better. Future work could be to fix the implementation \
    from the prototype notebook to make it perform faster.
    """

    Z = rng.standard_normal(size=(n, n))
    Q, r = np.linalg.qr(Z)
    d = np.diag(r)
    return Q * (d / np.abs(d)) @ Q
