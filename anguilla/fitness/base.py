"""This module contains abstract definitions for objective functions."""
from __future__ import annotations

import abc

import numpy as np


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
    n_objectives
        The number of objectives.
    eval_count
        Evaluation counter.

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
        self.n_objectives = 1
        self.eval_count = 0

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
        raise NotImplementedError()

    def __call__(self, x):
        """Compute the function value at a point(s)."""
        self.eval_count += 1
        if len(x.shape) > 1:
            self.eval_count += x.shape[0] - 1
            out = np.empty((x.shape[0],), dtype=float)
            self._eval(x, out)
            return out
        out = np.empty((1,), dtype=float)
        self._eval(x.reshape(1, x.shape[0]), out)
        return out[0]
