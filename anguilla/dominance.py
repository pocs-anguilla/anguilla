"""This module contains implementations of dominance operators."""
from typing import Optional, Union

import numpy as np


def fast_non_dominated_sort(
    ps: np.ndarray,
    target_n: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Rank the points using the Pareto dominance relation.

    Parameters
    ----------
    ps
        The set of objective points.
    target_n: optional
        The target number of points (e.g., for selection).
    out: optional
        Alternative output array.

    Returns
    -------
    np.ndarray: optional
        The ranks of the points (i.e. the i-th element \
        corresponds to the rank of the i-th point) with 1 being the best.
        Equivalently, it labels each objective point with its corresponding \
        front.
        If ``target_n`` is passed, sorts the minimum amount of fronts to \
        at least have the required target number of points. A value of zero \
        represents unranked points (with no front assigned).

    Notes
    -----
    Implements the algorithm presented in :cite:`2002:nsga-ii` and \
    used in the reference implementation :cite:`2008:shark`.
    It also implements the optional eager termination criterion \
    described in p. 5 of :cite:`2002:nsga-ii`.
    """
    size = len(ps)
    return_value = False
    if out is None:
        return_value = True
        out = np.empty(size)

    # Set of solutions that the solution p dominates
    s = np.empty((size, size), dtype=bool)
    for i in range(size):
        # Compute dominated points
        np.all(ps[i] < ps, axis=1, out=s[i])
    # Domination count
    n = np.sum(s, axis=0)
    front = n == 0
    out[:] = np.where(front, 1, 0)
    i = 1
    while np.any(front) and (target_n is None or np.sum(out != 0) < target_n):
        n -= np.sum(s[front], axis=0)
        i += 1
        np.logical_and(n == 0, out == 0, out=front)
        out[front] = i
    return out if return_value else None


def naive_non_dominated_sort(points: np.ndarray) -> np.ndarray:
    """Naive implementation of non-dominated sort.

    Notes
    -----
    Described in section III.A, pp. 183-184 of :cite:`2002:nsga-ii`, \
    this naive algorithm has a time complexity of O(M N^3). \
    It is provided for testing purposes solely.
    """
    size = len(points)
    max_level = len(points[0])
    rank = np.ones(size, dtype=int)
    level = 1
    while level <= max_level:
        next_level = level + 1
        for i in range(size):
            for j in range(size):
                if (
                    (i != j)
                    and (rank[i] == level)
                    and (rank[j] == level)
                    and np.all(points[i] < points[j])
                ):
                    rank[j] = next_level
        level = next_level
    return rank
