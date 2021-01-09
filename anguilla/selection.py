"""This module contains implementations of selection operators."""
from typing import Tuple

import numpy as np

from anguilla.dominance import fast_non_dominated_sort
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
    fronts, rank = fast_non_dominated_sort(points)

    current_size = len(fronts)
    current_front = fronts == rank
    current_front_size = np.sum(current_front)
    while current_size - current_front_size >= target_size:
        selected[current_front] = False
        current_size -= current_front_size
        rank -= 1
        current_front = fronts == rank
        current_front_size = np.sum(current_front)

    current_front = np.argwhere(current_front).flatten()
    contributions = indicator.contributions(points[current_front])

    # Remove the smallest contributions
    sorted_idx = np.argsort(contributions)
    selected[current_front[sorted_idx[: current_size - target_size]]] = False

    return selected, fronts
