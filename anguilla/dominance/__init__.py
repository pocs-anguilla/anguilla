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
    dominates_any = False
    dominates_all = True
    size = len(a)
    for i in range(size):
        dominates_any = dominates_any or a[i] < b[i]
        dominates_all = dominates_all and not (a[i] > b[i])
    return dominates_all and dominates_any


class NonDominatedSetKD:
    """Models a non-dominated set of k-D points."""

    def __init__(self):
        self.non_dominated_points = None

    def insert(self, points):
        if self.non_dominated_points is not None:
            union = np.vstack((self.non_dominated_points, points))
        else:
            union = np.array(points)

        ranks, _ = non_dominated_sort(union, 1)
        self.non_dominated_points = union[ranks == 1]

    @property
    def upper_bound(self):
        reference = np.max(self.non_dominated_points, axis=0)
        return reference


__all__ = [
    "non_dominated_sort",
    "dominates",
    "NonDominatedSet2D",
    "NonDominatedSetKD",
]
