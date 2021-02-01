"""This module contains operators related to Pareto dominance."""
from typing import Iterable

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


def union_upper_bound(
    *populations: Iterable[np.ndarray], translate_by: float = 0.0
):
    """Compute the upper bound of the union of non-dominated individuals \
    in a set of populations.

    Parameters
    ----------
    populations
        The populations of individuals to build the reference set with.
    translate_by: optional
        Number to translate the upper bound with.

    Returns
    -------
        The translated upper bound of the union of non-nominated individuals.

    Notes
    -----
    Based on the reference set described in Section 4.2, p. 490 \
    :cite:`2010:mo-cma-es`.
    """
    reference_set = np.vstack(populations)
    ranks, _ = non_dominated_sort(reference_set, 1)
    reference_set = reference_set[ranks == 1]
    return np.max(reference_set, axis=0) + translate_by


__all__ = [
    "non_dominated_sort",
    "dominates",
    "union_upper_bound",
]
