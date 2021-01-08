"""This module contains implementations of quality indicators."""
import abc
from typing import Optional, final

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
    def contributions(points: np.ndarray) -> np.ndarray:
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

    @abc.abstractmethod
    @final
    def __call__(points: np.ndarray) -> np.ndarray:
        """Compute the quality indicator.

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
        if self._reference is None:
            return hv.contributions(points, np.max(points, axis=0))
        return hv.contributions(points, self._reference)

    def __call__(self, points: np.ndarray) -> float:
        if self._reference is None:
            return hv.calculate(points, np.max(points, axis=0))
        return hv.calculate(points, self._reference)
