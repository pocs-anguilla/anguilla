"""Hypervolume algorithms."""

import numpy as np
from typing import Optional

from ._hypervolume import (
    hv2d_f8,
    hv3d_btree_f8,
    hv3d_rbtree_f8,
    hvc2d_f8,
    hvc3d_btree_f8,
    hvc3d_rbtree_f8,
)

__all__ = ["calculate", "contributions"]

# Optional
try:
    from ._shark_hypervolume import hvkd_f8 as shark_calculate
    from ._shark_hypervolume import hvckd_f8 as shark_contributions

    __all__.append("shark_calculate")
    __all__.append("shark_contributions")
    SHARK_BINDINGS_AVAILABLE = True
except ImportError:
    SHARK_BINDINGS_AVAILABLE = False


def calculate(
    ps: np.ndarray, ref_p: Optional[np.ndarray] = None, ds: str = "btree"
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
        if ds == "btree":
            return hv3d_btree_f8(ps, ref_p)
        elif ds == "rbtree":
            return hv3d_rbtree_f8(ps, ref_p)
        else:
            raise ValueError("ds: {}".format(ds))
    elif d > 3:
        if SHARK_BINDINGS_AVAILABLE:
            shark_calculate(ps, ref_p)
        else:
            raise NotImplementedError()
    else:
        raise ValueError("Input dimensionality can't be one.")
    return 0.0


def contributions(
    ps: np.ndarray, ref_p: Optional[np.ndarray] = None, ds: str = "btree"
) -> np.ndarray:
    """Compute the hypervolume contribution for a set of k-D points.

    Parameters
    ----------
    ps
        The set of mutually non-dominated points.
    ref_p: optional
        The reference point.
    ds: optional
        The data structure to use for implementing the sweeping structure.
        By default `btree` (B-tree), but can also be `rbtree` (Red-black tree).

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
        return hvc2d_f8(ps, ref_p)
    elif d == 3:
        if ds == "btree":
            return hvc3d_btree_f8(ps, ref_p)
        elif ds == "rbtree":
            return hvc3d_rbtree_f8(ps, ref_p)
        else:
            raise ValueError("ds: {}".format(ds))
    elif d > 3:
        if SHARK_BINDINGS_AVAILABLE:
            shark_contributions(ps, ref_p)
        else:
            raise NotImplementedError()
    else:
        raise ValueError("Input dimensionality can't be one.")
    return np.empty(0)


def contributions_naive(
    ps: np.ndarray, ref_p: Optional[np.ndarray] = None
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
    if ps.size == 0:
        return np.empty(0)

    if ref_p is None:
        ref_p = np.zeros_like(ps[0])

    contribution = np.zeros(len(ps))

    vol = calculate(ps, ref_p)
    for i in range(len(ps)):
        qs = np.delete(ps, i, 0)
        contribution[i] = vol - calculate(qs, ref_p)
    return contribution
