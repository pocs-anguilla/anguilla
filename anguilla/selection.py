"""This module contains implementations of selection operators."""
from typing import Tuple

import numpy as np

from anguilla.dominance import non_dominated_sort
from anguilla.indicators import Indicator


def indicator_selection(
    indicator: Indicator, points: np.ndarray, target_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform selection using a quality indicator.

    Parameters
    ----------
    indicator
        The indicator.
    points
        The objective points.
    target_size
        The number of objective points to select.

    Returns
    -------
    Tuple[np.ndarray]
        A flag array indicating the selected points and the ranks array.

    Notes
    -----
    Based on the `IndicatorBasedSelection` class from :cite:`2008:shark`.
    """
    selected = np.ones(len(points), dtype=bool)
    ranks, rank = non_dominated_sort(points)

    current_size = len(ranks)
    current_front = ranks == rank
    current_front_size = np.sum(current_front)
    while current_size - current_front_size >= target_size:
        selected[current_front] = False
        current_size -= current_front_size
        rank -= 1
        current_front = ranks == rank
        current_front_size = np.sum(current_front)

    current_front = np.argwhere(current_front).flatten()
    k = current_size - target_size
    least_contributors_idx = indicator.least_contributors(
        points[current_front], k
    )
    # Remove the smallest contributions
    selected[current_front[least_contributors_idx]] = False

    return selected, ranks
