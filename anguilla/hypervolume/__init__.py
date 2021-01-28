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

__all__ = ["calculate", "contributions"]

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
    ps: np.ndarray,
    ref_p: Optional[np.ndarray] = None,
    use_btree: bool = True,
) -> float:
    """Compute the exact hypervolume indicator for a set of k-D points.

    Parameters
    ----------
    ps
        The point set of mutually non-dominated points.
    ref_p: optional
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
    if len(ps.shape) == 1:
        ps = np.array([ps])
    d = len(ps[0])
    if d == 2:
        return hv2d_f8(ps, ref_p)
    elif d == 3:
        return hv3d_f8(ps, ref_p, use_btree)
    elif d > 3:
        if SHARK_BINDINGS_AVAILABLE:
            shark_calculate(ps, ref_p)
        else:
            raise NotImplementedError()
    else:
        raise ValueError("Input dimensionality can't be one.")
    return 0.0


def contributions(
    ps: np.ndarray,
    ref_p: Optional[np.ndarray] = None,
    use_btree: bool = True,
    non_dominated: bool = True,
) -> np.ndarray:
    """Compute the hypervolume contribution for a set of k-D points.

    Parameters
    ----------
    ps
        The set of points.
    ref_p: optional
        The reference point.
    use_btree
        Only relevant for the 3-D implementation.
        The data structure to use for implementing the sweeping structure.
        By default `btree` (B-tree), but can also be `rbtree` (Red-black tree).
    non_dominated
        Only relevant for the 2-D implementation.
        If true, selects a more efficiente algorithm to perform the computation.

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
        return hvc2d_f8(ps, ref_p, non_dominated)
    elif d == 3:
        return hvc3d_f8(ps, ref_p, use_btree)
    elif d > 3:
        if SHARK_BINDINGS_AVAILABLE:
            shark_contributions(ps, ref_p)
        else:
            raise NotImplementedError()
    else:
        raise ValueError("Input dimensionality can't be one.")
    return np.empty(0)


def contributions_naive(
    ps: np.ndarray,
    ref_p: Optional[np.ndarray] = None,
    duplicates_possible: bool = True,
) -> np.ndarray:
    """Compute the hypervolume contribution for a set of points.

    Parameters
    ----------
    ps
        The set of mutually non-dominated points.
    ref_p: optional
        The reference point. Otherwise assumed to be the origin.

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
    n = len(ps)

    if n == 0:
        return np.empty(0)

    if ref_p is None:
        ref_p = np.zeros_like(ps[0])

    vol = calculate(ps, ref_p)

    unique = []
    mappings = []

    sorted_idx = np.empty(0)
    if duplicates_possible:
        sorted_idx = np.argsort(ps[:, 0])
        ps = ps[sorted_idx, :]
        prev_index = 0
        prev_p = ps[prev_index]
        unique.append(prev_index)

        for index in range(1, n):
            p = ps[index]
            if np.array_equiv(prev_p, p):
                mappings.append((prev_index, index))
            else:
                unique.append(index)
            prev_index = index
            prev_p = p

        ps = ps[unique]

    tmp = np.zeros(len(ps))
    for i in range(len(ps)):
        qs = np.delete(ps, i, 0)
        tmp[i] = vol - calculate(qs, ref_p)

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
