"""This module contains abstract definitions related to objective functions."""
from __future__ import annotations

import abc
import typing

import numpy as np

Bounds = typing.Union[float, np.ndarray]
BoundsTuple = typing.Tuple[Bounds, Bounds]


class AbstractObjectiveFunction(metaclass=abc.ABCMeta):
    """An objective function.

    Parameters
    ----------
    n_dimensions
        The dimensionality of the search space.
    rng : optional
        A random number generator.

    Attributes
    ----------
    name
        The name of the function.
    n_objectives
        The number of objectives (i.e. dimensionality of the solution space).
    n_dimensions
        The dimensionality of the search space.
    evaluation_count
        Evaluation counter.

    Notes
    -----
    This class is based on the ``AbstractObjectiveFunction`` class \
        from :cite:`2008:shark`.

    Functions that need to update their internal state depending on \
    ``n_dimensions`` should override :meth:`_handle_dimensions_update`.
    """

    _rng: np.random.Generator
    _n_objectives: int
    _n_dimensions: int
    _evaluation_count: int
    _constraints_handler: typing.Optional[AbstractConstraintsHandler]

    def __init__(self, n_dimensions: int, rng: np.random.Generator = None):
        """Initialize the internal state of the objective function."""
        self._n_objectives = 1
        self._constraints_handler = None
        self._evaluation_count = 0
        self._n_dimensions = n_dimensions

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

        self._handle_dimensions_update()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def n_dimensions(self) -> int:
        return self._n_dimensions

    @n_dimensions.setter
    def n_dimensions(self, n_dimensions: int) -> None:
        self._n_dimensions = n_dimensions
        self._handle_dimensions_update()

    @property
    def n_objectives(self) -> int:
        return self._n_objectives

    @property
    def evaluation_count(self) -> int:
        return self._evaluation_count

    def random_points(
        self,
        m: int = 1,
        region_bounds: typing.Optional[BoundsTuple] = None,
    ) -> np.ndarray:
        """Compute random point(s) from the search space.

        Parameters
        ----------
            m
                The number of points.
            region_bounds: optional
                Initial region bounds.

        Returns
        -------
            np.ndarray
                The proposed point(s) with:

                * Standard normal distribution (default)
                * Uniform distribution (default if using constraints handler)
                * Uniform distribution within the provided region bounds \
                  (any constraints will be applied to the custom region)
        """
        if self._constraints_handler is not None:
            return self._constraints_handler.random_points(
                self._rng, m, region_bounds
            )
        n = self._n_dimensions
        shape = (m, n) if m > 1 else (n,)

        if region_bounds is not None:
            lower, upper = region_bounds
            return self._rng.uniform(lower, upper, size=shape)

        return self._rng.standard_normal(shape)

    def closest_feasible(self, x: np.ndarray) -> np.ndarray:
        """Compute the closest feasible point(s).

        Parameters
        ----------
        x
            The search point(s).

        Returns
        -------
        np.ndarray
            The closest feasible point(s).
        """
        if self._constraints_handler is not None:
            return self._constraints_handler.closest_feasible(x)
        return x

    @abc.abstractclassmethod
    def evaluate(self, x: np.ndarray) -> typing.Union[float, np.ndarray]:
        """Evaluate the function at the given search point.

        Parameters
        ----------
        x
            The search point.

        Returns
        -------
        typing.Union[float, np.ndarray]
            The value of the function at the search point.

        Raises
        ------
        ValueError
            Point has invalid shape.
        """
        raise NotImplementedError()

    def evaluate_multiple(
        self, x: np.ndarray
    ) -> typing.Union[float, np.ndarray]:
        """Evaluate the function at the given search points.

        Parameters
        ----------
        x
            The search points.

        Returns
        -------
        typing.Union[float, np.ndarray]
            The value of the function at the search points.

        Raises
        ------
        ValueError
            Points have invalid shape.
        """
        raise NotImplementedError()

    def _handle_dimensions_update(self) -> None:
        """Update any internal state dependent on the dimensionality \
        of the search space."""
        pass

    def _validate_point_shape(self, shape: typing.Tuple[int, ...]) -> None:
        """Validate the shape of given search point.

        Raises
        ------
        ValueError
            Point has invalid shape
        """
        if len(shape) > 1 or shape[0] > self._n_dimensions:
            raise ValueError("Point has invalid shape")

    def _validate_points_shape(self, shape: typing.Tuple[int, ...]) -> int:
        """Validate the shape of given search points.

        Returns
        -------
        int
            Index of the shape corresponding to the point(s) dimension.

        Raises
        ------
        ValueError
            Point(s) have invalid shape
        """
        axis = 1 if len(shape) > 1 else 0
        if shape[axis] > self._n_dimensions:
            raise ValueError("Point(s) have invalid shape")
        return axis

    def __call__(self, x: np.ndarray) -> typing.Union[float, np.ndarray]:
        """Syntactic sugar for :meth:`evaluate`."""
        return self.evaluate(x)

    def __repr__(self) -> str:
        """Return the name of the function."""
        return self.name


class AbstractConstraintsHandler(metaclass=abc.ABCMeta):
    """An abstract constraints handler.

    Notes
    -----
    This class is based on the ``AbstractConstraintsHandler`` class \
        from :cite:`2008:shark`.
    """

    @abc.abstractmethod
    def closest_feasible(self, x: np.ndarray) -> np.ndarray:
        """Compute the closest feasible point."""
        raise NotImplementedError()

    @abc.abstractmethod
    def random_points(
        self,
        rng: np.random.Generator,
        m: int = 1,
        region_bounds: typing.Optional[BoundsTuple] = None,
    ) -> np.ndarray:
        """Generate random point(s) compatible with the constraints.

        Parameters
        ----------
            rng
                A random number generator
            m: optional
                The number of point(s) to generate.
            region_bounds: optional
                The region from which to generate random point(s).
        """
        raise NotImplementedError()
