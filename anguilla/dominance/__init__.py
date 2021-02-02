"""This module contains operators related to Pareto dominance."""
import numpy as np

from ._dominance import non_dominated_sort_f8 as non_dominated_sort


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


__all__ = [
    "non_dominated_sort",
    "dominates",
]
