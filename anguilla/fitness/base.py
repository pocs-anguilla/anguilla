"""This module contains abstract definitions for objective functions."""
from __future__ import annotations

import abc
from abc import abstractproperty
import typing

import numpy as np
import numpy.typing as npt


class AbstractObjectiveFunction(metaclass=abc.ABCMeta):  # WIP
    """An objective function.

    Parameters
    ----------
    eval:
        A generalized ufunc implementing the function.
    rng : optional
        A random number generator.

    Attributes
    ----------
        name
            The name of the function.

    Notes
    -----
    This class is a Python port of the AbstractObjectiveFunction class \
        from :cite:`2008:shark`.
    
    See:

        * https://numpy.org/doc/stable/reference/ufuncs.html
    """

    name: str

    def __init__(self, eval, rng=None):
        """Initialize the objective function."""
        self._eval = eval

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

    @abc.abstractclassmethod
    def propose_initial_point(self, n: int, m: int = 1) -> np.ndarray:
        """Propose an initial point or points.

        Parameters
        ----------
            n
                The dimension of the point.
            m
                The number of points.

        Returns
        -------
            np.ndarray
                The proposed point.

        Notes
        -----
            Optional only. May not be relevant/possible for all functions.

        """
        raise NotImplemented()

    def __call__(self, x):
        """Compute the function value at a point(s)."""
        if len(x.shape) > 1:
            out = np.empty((x.shape[0],), dtype=float)
            self._eval(x, out)
            return out
        else:
            out = np.empty((1,), dtype=float)
            self._eval(x.reshape(1, x.shape[0]), out)
            return out[0]


class AbstractObjectiveFunctionWithInputDomain(AbstractObjectiveFunction):
    """An objective function that defines a commonly used \
       input domain.

    Attributes
    ----------
        x_min
            The lower bound of the input domain.
        x_max
            The upper bound of the input domain.

    Notes
    -----
        The common input domain can be useful for plotting.
    """

    x_min: float
    x_max: float
