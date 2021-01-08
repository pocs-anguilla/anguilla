"""This module contains implementations of constraint handlers."""
import typing

import numpy as np

from anguilla.fitness.base import (
    ConstraintsHandler,
    Bounds,
    BoundsTuple,
)


class BoxConstraintsHandler(ConstraintsHandler):
    """A box constraints handler.

    Parameters
    ----------
        n_dimensions: The dimensionality of the search space.
        bounds: The bounds defining the box constraints.

    Raises
    ------
    ValueError
        The provided bounds have incorrect shape.
    """

    _n_dimensions: int
    _lower_bounds: Bounds
    _upper_bounds: Bounds

    def __init__(self, n_dimensions: int, bounds: BoundsTuple) -> None:
        self._n_dimensions = n_dimensions
        if (
            not isinstance(bounds[0], float)
            and bounds[0].shape[0] != self._n_dimensions
        ):
            raise ValueError("Lower bounds have invalid shape")
        if (
            not isinstance(bounds[1], float)
            and bounds[1].shape[0] != self._n_dimensions
        ):
            raise ValueError("Upper bounds have invalid shape")
        self._lower_bounds, self._upper_bounds = bounds

    @property
    def n_dimensions(self) -> int:
        return self._n_dimensions

    @property
    def bounds(self) -> BoundsTuple:
        return self._lower_bounds, self._upper_bounds

    @bounds.setter
    def bounds(self, bounds: BoundsTuple) -> None:
        self._lower_bounds, self._upper_bounds = bounds

    def random_points(
        self,
        rng: np.random.Generator,
        m: int = 1,
        region_bounds: typing.Optional[BoundsTuple] = None,
    ) -> np.ndarray:
        n = self._n_dimensions
        shape = (m, n) if m > 1 else (n,)
        # Use custom region (within the box constraints)
        if region_bounds is not None:
            lower, upper = region_bounds
            lower = np.maximum(lower, self._lower_bounds)
            upper = np.minimum(upper, self._upper_bounds)
            return rng.uniform(lower, upper, size=shape)
        # Use the bounds from the box constraints as the region
        return rng.uniform(self._lower_bounds, self._upper_bounds, size=shape)

    def closest_feasible(self, x: np.ndarray) -> np.ndarray:
        """Compute the closest feasible point.

        Notes
        -----
        Implementation uses the definition from p. 12 of \
        :cite:`2007:mo-cma-es`.
        """
        return np.minimum(
            np.maximum(x, self._lower_bounds), self._upper_bounds
        )
