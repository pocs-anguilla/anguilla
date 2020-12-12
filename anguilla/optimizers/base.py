"""This module contains abstract classes modeling optimizers."""
from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


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

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the name of the optimizer."""
        raise NotImplementedError()

    def __init__(self, initial_point: np.ndarray) -> None:
        """Initialize the optimizer."""
        self._initial_point = initial_point
        self._fevals: int = 0

    @abc.abstractmethod
    def ask(self) -> np.ndarray:
        """Compute a new candidate solution.

        Returns
        -------
        np.ndarray
            The new candidate solution.
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    @abc.abstractmethod
    def stop(self) -> bool:
        """Check for stopping conditions."""
        raise NotImplementedError()

    @abc.abstractmethod
    def ask(self) -> np.ndarray:
        """Compute a new candidate solution.

        Returns
        -------
        np.ndarray
            The new candidate solution.
        """
        raise NotImplementedError()

    def optimize(self) -> AbstractOptimizer:
        """Run the optimization.

        Returns
        -------
        AbstractOptimizer
            The optimizer instance.
        """
        raise NotImplementedError()
