"""This module contains implementations of quality indicators."""
import abc
from typing import Optional

try:
    from typing import final
except ImportError:
    from typing_extensions import final

import numpy as np

import anguilla.hypervolume as hv


class Indicator(metaclass=abc.ABCMeta):
    """Define interface of a quality indicator."""

    @property
    @abc.abstractmethod
    @final
    def name(self) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    @final
    def contributions(self, points: np.ndarray) -> np.ndarray:
        """Compute the contributions of each point.

        Parameters
        ----------
        points
            The objective points.

        Result
        ------
        np.ndarray
            The contributions.
        """
        raise NotImplementedError()

    def least_contributors(self, points, k):
        """Compute the least contributors indices.

        Parameters
        ----------
        points
            The objective points.
        k
            The number of least contributors to determine.

        Result
        ------
        np.ndarray
            The least contributors' indices.

        Notes
        -----
        The algorithm is described as part of Lemma 1, page 11, of :cite:`2007:mo-cma-es`.
        """
        indices = []
        n = len(points)
        active_points = points.copy()
        active_indices = np.arange(n)
        for _ in range(k):
            contribs = self.contributions(active_points)
            index = np.argsort(contribs)[0]
            active_points = np.delete(active_points, index, 0)
            indices.append(active_indices[index])
            active_indices = np.delete(active_indices, index)
        return np.array(indices)

    @abc.abstractmethod
    @final
    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Compute the indicator value.

        Parameters
        ----------
        points
            The objective points.

        Result
        ------
        np.ndarray
            The value of the quality indicator.
        """
        raise NotImplementedError()


class HypervolumeIndicator(Indicator):
    """The hypervolume indicator.

    Parameters
    ----------
    reference
        Reference point.
    """

    def __init__(self, reference: Optional[np.ndarray] = None) -> None:
        self._reference = reference

    @property
    def name(self) -> str:
        return "Hypervolume"

    @property
    def reference(self) -> Optional[np.ndarray]:
        if self._reference is not None:
            return np.copy(self._reference)
        return None

    @reference.setter
    def reference(self, reference: np.ndarray) -> None:
        self._reference = reference

    def contributions(self, points: np.ndarray) -> np.ndarray:
        return hv.contributions(points, self._reference)

    def __call__(self, points: np.ndarray) -> float:
        if self._reference is None:
            return hv.calculate(points, np.max(points, axis=0))
        return hv.calculate(points, self._reference)
