"""This module contains implementations of dominance operators."""
from typing import Optional, Tuple

import numpy as np

DominationSortResult = Tuple[np.ndarray, int]


def fast_non_dominated_sort(
    points: np.ndarray,
    target_size: Optional[int] = None,
) -> DominationSortResult:
    """front the points using the Pareto dominance relation.

    Parameters
    ----------
    points
        The set of objective points.
    target_size: optional
        The target number of points (e.g., for selection).

    Returns
    -------
    DominationSortResult
        The array of ranks of each point and the maximum rank.

    Notes
    -----
    Implements the algorithm presented in :cite:`2002:nsga-ii` and \
    used in the reference implementation :cite:`2008:shark`.
    It also implements the optional eager termination criterion \
    described in p. 5 of :cite:`2002:nsga-ii`, unranked points have \
    a value of zero.
    """
    size = len(points)
    out = np.empty(size, dtype=int)

    # Set of solutions that the solution p dominates
    s = np.empty((size, size), dtype=bool)
    for i in range(size):
        # Compute dominated points
        np.all(points[i] < points, axis=1, out=s[i])
    # Domination count
    n = np.sum(s, axis=0)
    front = n == 0
    out[:] = np.where(front, 1, 0)
    i = 1
    while np.any(front) and (
        target_size is None or np.sum(out != 0) < target_size
    ):
        n -= np.sum(s[front], axis=0)
        i += 1
        np.logical_and(n == 0, out == 0, out=front)
        out[front] = i
    return out, i


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
    front = np.ones(size, dtype=int)
    level = 1
    while level <= max_level:
        next_level = level + 1
        for i in range(size):
            for j in range(size):
                if (
                    (i != j)
                    and (front[i] == level)
                    and (front[j] == level)
                    and np.all(points[i] < points[j])
                ):
                    front[j] = next_level
        level = next_level
    return front
