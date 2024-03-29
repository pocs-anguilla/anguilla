"""This module contains abstract definitions related to objective functions."""
import abc
from typing import Optional, Union, Tuple

import numpy as np

Bounds = Union[float, np.ndarray]
BoundsTuple = Tuple[Bounds, Bounds]


class ConstraintsHandler(metaclass=abc.ABCMeta):
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
        region_bounds: Optional[BoundsTuple] = None,
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

        Returns
        -------
        np.ndarray
            The random points.
        """
        raise NotImplementedError()


class ObjectiveFunction(metaclass=abc.ABCMeta):
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
    _constraints_handler: Optional[ConstraintsHandler]
    _has_scalable_dimensions: bool = True
    _has_scalable_objectives: bool = False
    _has_known_pareto_front: bool = False
    _has_continuous_pareto_front: bool = True

    def __init__(
        self,
        n_dimensions: int = 2,
        n_objectives: int = 1,
        rng: np.random.Generator = None,
    ):
        """Initialize the internal state of the objective function."""
        self._n_objectives = n_objectives
        self._constraints_handler = None
        self._evaluation_count = 0
        self._n_dimensions = n_dimensions

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

        self._post_update_n_dimensions()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the name of the objective function."""
        raise NotImplementedError()

    @property
    def qualified_name(self) -> str:
        """Return the qualified name of the objective function."""
        return self.name

    @property
    def n_dimensions(self) -> int:
        """Return th number of dimensions."""
        return self._n_dimensions

    @n_dimensions.setter
    def n_dimensions(self, n_dimensions: int) -> None:
        """Return the update number of dimensions.

        Raises
        ------
        ValueError
            The provided value is not valid.
        """
        if self._has_scalable_dimensions:
            self._pre_update_n_dimensions(n_dimensions)
            self._n_dimensions = n_dimensions
            self._post_update_n_dimensions()

    @property
    def has_scalable_dimensions(self):
        """Return True if number of dimensions can be scaled."""
        return self._has_scalable_dimensions

    @property
    def n_objectives(self) -> int:
        """Return the number of objectives."""
        return self._n_objectives

    @n_objectives.setter
    def n_objectives(self, n_objectives: int) -> None:
        """Update the number of objectives.

        Raises
        ------
        ValueError
            The provided value is not valid.
        AttributeError
            The value cannot be changed.
        """
        if self._has_scalable_objectives:
            self._pre_update_n_objectives(n_objectives)
            self._n_objectives = n_objectives
            self._post_update_n_objectives()

    @property
    def has_scalable_objectives(self):
        """Return True if number of objectives can be scaled."""
        return self._has_scalable_objectives

    @property
    def has_known_pareto_front(self):
        """Return True if the Pareto front is known and implemented."""
        return self._has_known_pareto_front

    @property
    def has_continuous_pareto_front(self):
        """Return True if the Pareto front is continuous."""
        return self._has_continuous_pareto_front

    @property
    def evaluation_count(self) -> int:
        """Return the evaluation count."""
        return self._evaluation_count

    @property
    def has_constraints(self) -> bool:
        """Return True if the function has constraints."""
        return self._constraints_handler is not None

    def random_points(
        self,
        m: int = 1,
        region_bounds: Optional[BoundsTuple] = None,
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
        shape = (m, self._n_dimensions)

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

    @abc.abstractmethod
    def evaluate(
        self,
        xs: np.ndarray,
        count: bool = True,
    ) -> np.ndarray:
        """Compute the fitness values.

        Parameters
        ----------
        xs
            The search points.
        count: optional
            Update the evaluation count.

        Return
        ------
        np.ndarray
            The fitness values.
        """
        raise NotImplementedError()

    def _pre_evaluate(self, xs: np.ndarray, count: bool = True) -> None:
        if xs.ndim != 2:
            msg = "Points array must have 2 dimensions. Got {}.".format(
                xs.ndim
            )
            raise ValueError(msg)
        if xs.shape[1] != self._n_dimensions:
            msg = "Search point must have {} dimensions. Got {}.".format(
                self._n_dimensions, xs.shape[1]
            )
            raise ValueError(msg)
        if count:
            self._evaluation_count += xs.shape[0]

    def evaluate_with_constraints(
        self,
        xs: np.ndarray,
        count: bool = True,
    ) -> np.ndarray:
        """Compute the fitness projecting points to their closest feasible.

        Parameters
        ----------
        xs
            The search points.
        count: optional
            Update the evaluation count.

        Return
        ------
        np.ndarray
            The fitness values.
        """
        return self.evaluate(self.closest_feasible(xs), count=count)

    def evaluate_with_penalty(
        self,
        xs: np.ndarray,
        count: bool = True,
        penalty: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the fitness projecting points to their closest feasible \
        and apply a penalty.

        Parameters
        ----------
        xs
            The search point(s).
        count: optional
            Update evaluation counter.
        penalty: optional
            The penalty factor.

        Return
        ------
        Tuple[np.ndarray, np.ndarray]
            The unpenalized and penalized fitness values, respectively.

        Notes
        -----
        Implements penalized evaluation as described in \
        p. 12 of :cite:`2007:mo-cma-es`.
        """
        feasible_xs = self.closest_feasible(xs)
        feasible_ys = self.evaluate(feasible_xs, count)
        tmp = xs - feasible_xs
        penalized_ys = feasible_ys + np.expand_dims(
            penalty * np.sum(tmp * tmp, axis=1), axis=1
        )
        return feasible_ys, penalized_ys

    def pareto_front(self, num: int = 50) -> np.ndarray:
        """Return the true Pareto front.

        Parameters
        ----------
        num
            Number of samples.

        Returns
        -------
        np.ndarray
            The true Pareto front.

        Raises
        ------
        NotImplementedError
            The function is not implemented.
        RuntimeError
            The function is not supported.

        Notes
        -----
        Only applicable to multi-objective functions.
        """
        if self._n_objectives == 1:
            raise RuntimeError("Unsupported for single-objective functions")
        raise NotImplementedError()

    def _post_update_n_dimensions(self) -> None:
        """Apply any required logic after updating the dimensions \
        (e.g., recompute some internal state)."""

    def _pre_update_n_dimensions(self, n_dimensions: int) -> None:
        """Apply any required logic before updating the dimensions \
        (e.g. validation).

        Raises
        ------
        ValueError
            The provided value is not valid.
        """
        if n_dimensions < 1:
            raise ValueError("Invalid dimensions")

    def _post_update_n_objectives(self) -> None:
        """Apply any required logic after updating the number of objectives \
        (e.g., recompute some internal state)."""

    def _pre_update_n_objectives(self, n_objectives: int) -> None:
        """Apply any required logic before updating the number of objectives \
        (e.g. validation).

        Raises
        ------
        ValueError
            The provided value is not valid.
        AttributeError
            The value cannot be changed.
        """
        raise AttributeError("Number of objectives cannot be changed.")

    def __call__(
        self,
        xs: np.ndarray,
        count: bool = True,
        penalty: Optional[float] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Syntactic sugar to call either :meth:`evaluate` or \
            :meth:`evaluate_with_constraints`.

        Parameters
        ----------
        xs
            The search point(s).
        count: optional
            Update the evaluation count.
        penalty: optional
            Penalty factor.

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            The fitness.
        """
        if penalty is not None:
            return self.evaluate_with_penalty(xs, count=count, penalty=penalty)
        return self.evaluate(xs, count=count)

    def __repr__(self) -> str:
        """Return the name of the function."""
        return self.name


__all__ = [
    "ConstraintsHandler",
    "ObjectiveFunction",
]
