"""Hypervolume module."""

import numpy as np

from typing import Optional

from anguilla.hypervolume.exact import (
    calculate_2d,
    calculate_3d,
    contributions_2d_naive,
    contributions_3d,
)

__all__ = ["calculate", "contributions", "least_contributors"]


def calculate(ps: np.ndarray, ref_p: Optional[np.ndarray] = None) -> float:
    """Compute the exact hypervolume indicator for a set of k-D points.

    Parameters
    ----------
    ps
        The point set of mutually non-dominated points.
    ref_p
        The reference point. Otherwise assumed to be the origin.

    Raises
    ------
    ValueError
        The input's dimensionality is 1.

    Notes
    -----
    Chooses which of the available implementations to use, based on the \
    dimensionality of the points, as done in :cite:`2008:shark`.

    """
    if len(ps.shape) == 1:
        ps = np.array([ps])
    d = len(ps[0])
    if d == 2:
        return calculate_2d(ps, ref_p)
    elif d == 3:
        return calculate_3d(ps, ref_p)
    elif d > 3:
        raise NotImplementedError()
    else:
        raise ValueError("Input dimensionality can't be one.")


def contributions(
    ps: np.ndarray, ref_p: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute the hypervolume contribution for a set of k-D points.

    Parameters
    ----------
    ps
        The set of mutually non-dominated points.
    ref_p
        The reference point.

    Returns
    -------
    np.ndarray
        The hypervolume contribution of each point, respectively.

    Raises
    ------
    ValueError
        The input's dimensionality is 1.

    Notes
    -----
    Chooses which of the available implementations to use, based on the \
    dimensionality of the points, as done in :cite:`2008:shark`.
    """
    d = len(ps[0])
    if d == 2:
        return contributions_2d_naive(ps, ref_p)
    elif d == 3:
        return contributions_3d(ps, ref_p)
    elif d > 3:
        raise NotImplementedError()
    else:
        raise ValueError("Input dimensionality can't be one.")
