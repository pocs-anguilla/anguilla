"""This module contains operators related to Pareto dominance."""
import numpy as np

from ._dominance import non_dominated_sort_f8 as non_dominated_sort
from ._dominance import NonDominatedSet2D


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Determine if a point dominates another one.

    Paramaters
    ----------
    a
        Objective point that dominates.
    b
        Objective point that is dominated.

    Returns
    -------
    bool
        True if a dominates b.

    Raises
    ------
    ValueError
        The points have different size.
    """
    if a.size != b.size:
        raise ValueError("Input arrays have different sizes.")
    any = False
    all = True
    size = len(a)
    for i in range(size):
        any = any or a[i] < b[i]
        all = all and not (a[i] > b[i])
    return all and any


class NonDominatedSetKD:
    """Models a non-dominated set of k-D points."""

    def __init__(self):
        self.union = []

    def insert(self, points):
        self.union.append(points)

    @property
    def upper_bound(self):
        points = np.vstack(self.union)
        ranks, _ = non_dominated_sort(points, 1)
        nondominated = points[ranks == 1]
        reference = np.max(nondominated, axis=0)
        return reference


__all__ = [
    "non_dominated_sort",
    "dominates",
    "NonDominatedSet2D",
    "NonDominatedSetKD",
]
