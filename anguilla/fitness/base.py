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
    _scalable_dimensions: bool = True
    _scalable_objectives: bool = False

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
        if self._scalable_dimensions:
            self._pre_update_n_dimensions(n_dimensions)
            self._n_dimensions = n_dimensions
            self._post_update_n_dimensions()

    @property
    def scalable_dimensions(self):
        """Return True if number of dimensions can be scaled."""
        return self._scalable_dimensions

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
        if self._scalable_objectives:
            self._pre_update_n_objectives(n_objectives)
            self._n_objectives = n_objectives
            self._post_update_n_objectives()

    @property
    def scalable_objectives(self):
        """Return True if number of objectives can be scaled."""
        return self._scalable_objectives

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

    @abc.abstractmethod
    def evaluate_single(self, x: np.ndarray) -> Union[float, np.ndarray]:
        """Evaluate the function at the given search point.

        Parameters
        ----------
        x
            The search point.

        Returns
        -------
        Union[float, np.ndarray]
            The value of the function at the search point.

        Raises
        ------
        ValueError
            Point has invalid shape.

        Notes
        -----
        The implementation should be specialized to evaluate a single point
        efficiently.
        """
        raise NotImplementedError()

    def evaluate_multiple(self, xs: np.ndarray) -> Union[float, np.ndarray]:
        """Evaluate the function at the given search point(s).

        Parameters
        ----------
        xs
            The search point(s).

        Returns
        -------
        Union[float, np.ndarray]
            The value of the function at the search points.

        Raises
        ------
        ValueError
            Points have invalid shape.

        Notes
        -----
        The implementation should be specialized to evaluate multiple points
        efficiently. A default non-specialized implementation is provided \
        as default. Implementations should override this method.
        """
        xs = self._pre_evaluate_multiple(xs, False)
        values = np.array([self.evaluate_single(x) for x in xs])
        return values if len(values) > 1 else values[0]

    def evaluate_with_constraints(
        self, xs: np.ndarray
    ) -> Union[float, np.ndarray]:
        """Compute the fitness projecting points to their closest feasible.

        Parameters
        ----------
        xs
            The search point(s).

        Return
        ------
        Union[float, np.ndarray]
            The fitness value(s).
        """
        return self.__call__(self.closest_feasible(xs))

    def evaluate_with_penalty(
        self, xs: np.ndarray, penalty: float = 10e-6
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Compute the fitness projecting points to their closest feasible \
        and apply a penalty.

        Parameters
        ----------
        xs
            The search point(s).

        Return
        ------
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]
            The unpenalized and penalized fitness value(s), respectively.

        Notes
        -----
        Implements penalized evaluation as described in \
        p. 12 of :cite:`2007:mo-cma-es`.
        """
        feasible_xs = self.closest_feasible(xs)
        tmp = xs - feasible_xs
        ys = self.__call__(feasible_xs)
        return ys, ys + penalty * np.sum(tmp * tmp)

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
            Unsupported or not implemented.

        Notes
        -----
        Only applicable to multi-objective functions.
        """
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

    def _pre_evaluate_single(self, x: np.ndarray, count: bool = True) -> None:
        """Validate the shape of given search point and update evaluation \
        counter.

        Parameter
        ---------
        x
            The search point.

        Raises
        ------
        ValueError
            Point has invalid shape
        """
        if len(x.shape) > 1 or len(x) != self._n_dimensions:
            raise ValueError("Point has invalid shape")
        if count:
            self._evaluation_count += 1

    def _pre_evaluate_multiple(
        self, xs: np.ndarray, count: bool = True
    ) -> np.ndarray:
        """Validate the shape of given search point(s) and update evaluation \
            counter.

        Parameters
        ----------
        xs
            The search point(s).

        Returns
        -------
        xs
            A new view object if possible or a copy (``np.reshape`` behavior).

        Raises
        ------
        ValueError
            Point(s) have invalid shape

        """
        if len(xs.shape) < 2:
            xs = np.reshape(xs, (1, len(xs)))

        if xs.shape[1] != self._n_dimensions:
            raise ValueError("Point(s) have invalid shape")
        if count:
            self._evaluation_count += len(xs)

        return xs

    def __call__(self, x: np.ndarray) -> Union[float, np.ndarray]:
        """Syntactic sugar to call either :meth:`evaluate` \
        or :meth:`evaluate_multiple` depending on the shape of the array.

        Parameters
        ----------
        x
            The search point(s).
        """
        if len(x.shape) > 1:
            return self.evaluate_multiple(x)
        return self.evaluate_single(x)

    def __repr__(self) -> str:
        """Return the name of the function."""
        return self.name
