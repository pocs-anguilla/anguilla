"""This module contains utilitary/auxilliary functions."""
import math
import functools

from typing import Any, Optional, Callable, Tuple, List


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
