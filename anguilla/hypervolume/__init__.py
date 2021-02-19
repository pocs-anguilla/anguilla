"""Hypervolume algorithms."""

import os
import platform

import numpy as np
from typing import Optional

from ._hypervolume import (
    hv2d_f8,
    hv3d_f8,
    hvc2d_f8,
    hvc3d_f8,
)

from .exact import hvkd as prototype_hvkd

__all__ = ["calculate", "contributions", "contributions_naive"]

# Optional
try:
    from ._shark_hypervolume import hvkd_f8 as shark_calculate

    __all__.append("shark_calculate")

    from ._shark_hypervolume import hvckd_f8 as shark_contributions

    __all__.append("shark_contributions")

    SHARK_BINDINGS_AVAILABLE = True
except ImportError:
    SHARK_BINDINGS_AVAILABLE = False


def calculate(
    points: np.ndarray,
    reference: Optional[np.ndarray] = None,
    ignore_dominated: bool = False,
    use_btree: bool = True,
) -> float:
    """Compute the exact hypervolume indicator for a set of k-D points.

    Parameters
    ----------
    points
        The point set of mutually non-dominated points.
    reference: optional
        The reference point. Otherwise assumed to be the origin.
    ds: optional
        The data structure to use for implementing the sweeping structure.
        By default `btree` (B-tree), but can also be `rbtree` (Red-black tree).

    Raises
    ------
    ValueError
        The input's dimensionality is 1.

    Notes
    -----
    Chooses which of the available implementations to use, based on the \
    dimensionality of the points, as done in :cite:`2008:shark`.

    """
    if len(points.shape) == 1:
        points = np.array([points])
    d = len(points[0])
    if d == 2:
        return hv2d_f8(points, reference, ignore_dominated)
    elif d == 3:
        return hv3d_f8(points, reference, ignore_dominated, use_btree)
    elif d > 3:
        if SHARK_BINDINGS_AVAILABLE:
            return shark_calculate(points, reference)
        else:
            return prototype_hvkd(points, reference)
    else:
        raise ValueError("Input dimensionality can't be one.")


def contributions(
    points: np.ndarray,
    reference: Optional[np.ndarray] = None,
    use_btree: bool = True,
    non_dominated: bool = True,
    prefer_extrema: bool = False,
) -> np.ndarray:
    """Compute the hypervolume contribution for a set of k-D points.

    Parameters
    ----------
    points
        The set of points.
    reference: optional
        The reference point.
    use_btree
        Only relevant for the 3-D implementation.
        The data structure to use for implementing the sweeping structure.
        By default `btree` (B-tree), but can also be `rbtree` (Red-black tree).
    non_dominated
        Only relevant for the 2-D implementation.
        If true, selects a more efficiente algorithm to perform the \
        computation.
    prefer_extrema
        Whether to give preference to extremum points by assigning them \
        infinite contribution.

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
    d = len(points[0])
    if d == 2:
        return hvc2d_f8(points, reference, non_dominated)
    elif d == 3:
        return hvc3d_f8(points, reference, use_btree, prefer_extrema)
    elif d > 3:
        if SHARK_BINDINGS_AVAILABLE:
            return shark_contributions(points, reference)
        else:
            return NotImplementedError()
    else:
        raise ValueError("Input dimensionality can't be one.")


def contributions_naive(
    points: np.ndarray,
    reference: Optional[np.ndarray] = None,
    duplicates_possible: bool = False,
) -> np.ndarray:
    """Compute the hypervolume contribution for a set of points.

    Parameters
    ----------
    points
        The set of points.
    reference: optional
        The reference point. Otherwise assumed to be the component-wise \
        maximum.

    Returns
    -------
    np.ndarray
        The hypervolume contribution of each point, respectively.

    Notes
    -----
    The brute-force approach to computing the hypervolume contributions.
    Uses the available hypervolume calculation function to compute the \
    contribution of each point using its definition \
    (p. 4 of :cite:`2020:hypervolume`):

    .. math::
        H(p, S) = H(S \\cup {p}) - H(S \\ {p})

    Provided only for testing the other implementation, as done in \
    :cite:`2008:shark`.
    """
    n = len(points)

    if n == 0:
        return np.empty(0)

    if reference is None:
        reference = np.max(points, axis=0)

    vol = calculate(points, reference)

    unique = []
    mappings = []

    sorted_idx = np.empty(0)
    if duplicates_possible:
        sorted_idx = np.argsort(points[:, 0])
        points = points[sorted_idx, :]
        prev_index = 0
        prev_p = points[prev_index]
        unique.append(prev_index)

        for index in range(1, n):
            p = points[index]
            if np.array_equiv(prev_p, p):
                mappings.append((prev_index, index))
            else:
                unique.append(index)
            prev_index = index
            prev_p = p

        points = points[unique]

    tmp = np.zeros(len(points))
    for i in range(len(points)):
        qs = np.delete(points, i, 0)
        tmp[i] = vol - calculate(qs, reference)

    if duplicates_possible:
        contribution = np.zeros(n)
        for index, contrib in zip(unique, tmp):
            contribution[index] = contrib
        for orig, duplicate in mappings:
            contribution[duplicate] = contribution[orig]
        reverse_idx = np.argsort(sorted_idx)
        return contribution[reverse_idx]
    else:
        return tmp
