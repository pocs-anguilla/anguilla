"""This module contains abstract classes modeling optimizers."""
from __future__ import annotations

import abc

import numpy as np
import numpy.typing as npt


class AbstractOptimizer(metaclass=abc.ABCMeta):
    """An abstract optimizer.

    Parameters
    ----------
    initial_point
        The initial search point.

    Notes
    -----
    It uses the ask-and-tell pattern presented in \
        :cite:`2013:oo-optimizers` and implemented by :cite:`2019:pycma`.
    """

    _initial_point: np.ndarray

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        """Return the name of the optimizer."""
        pass

    def __init__(self, initial_point: np.ndarray) -> None:
        """Initialize the optimizer."""
        self._initial_point = initial_point
        self._fevals: int = 0

    @abc.abstractmethod
    def reset(self) -> None:
        """Re-initialize the optimizer."""
        pass

    @abc.abstractmethod
    def ask(self) -> np.ndarray:
        """Compute a new candidate solution.

        Returns
        -------
        np.ndarray
            The new candidate solution.
        """
        pass

    @abc.abstractmethod
    def tell(self, solutions: np.ndarray, fvalues: np.ndarray) -> None:
        """Feed the optimizer with objective function information.

        Parameters
        ----------
        solutions
            The candidate solutions.
        fvalues
            The function value of the candidate solutions.
        """
        pass

    @abc.abstractmethod
    def stop(self) -> bool:
        """Pending."""
        pass

    def optimize(self) -> AbstractOptimizer:
        """Run the optimization.

        Returns
        -------
        AbstractOptimizer
            The optimizer instance.
        """
        pass
