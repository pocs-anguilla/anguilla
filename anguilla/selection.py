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
    selected = np.zeros(len(points), dtype=bool)
    ranks, _ = non_dominated_sort(points)

    n_pending_select = target_size
    rank = 1
    pending = True
    while pending:
        front = ranks == rank
        n_front = np.sum(front)
        diff = n_pending_select - n_front
        if diff > 0:
            # We must select all individuals in the current front.
            selected[front] = True
            n_pending_select = diff
            rank += 1
        elif diff == 0:
            # All pending individuals are exactly in this front.
            selected[front] = True
            pending = False
        else:
            # We select the rest of pending individuals among individuals
            # in the current front by discarding the least contributors.
            idx = np.arange(len(ranks))[front]
            least_contributors = indicator.least_contributors(
                points[front], -diff,
            )
            selected[front] = True
            selected[idx[least_contributors]] = False
            pending = False

    return selected, ranks
