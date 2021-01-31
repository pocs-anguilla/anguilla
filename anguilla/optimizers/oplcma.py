"""This module contains implementations related to the OnePlusLambda-CMA-ES \
    algorithm for real-valued single-objective optimization (MOO)."""
import dataclasses
import math
from typing import Optional, Union

import numpy as np

from anguilla.optimizers.base import (
    Optimizer,
    OptimizerParameters,
    OptimizerSolution,
    OptimizerStoppingConditions,
)


@dataclasses.dataclass
class OPLParameters(OptimizerParameters):
    """Parameters for OnePlusLambda-CMA-ES.

    Parameters
    ----------
    n_dimensions
        Dimensionality of the search space.
    initial_step_size
        Initial step size.
    n_offspring: optional
        Number of offspring per parent.
    d: optional
        Step size damping parameter.
    p_target_succ: optional
        Target success probability.
    c_p: optional
        Success rate averaging parameter.
    p_threshold: optional
        Smoothed success rate threshold.
    c_c: optional
        Evolution path learning rate.
    c_cov: optional
        Covariance matrix learning rate.

    Notes
    -----
    Implements the default values defined in Table 1, p. 5 \
    :cite:`2007:mo-cma-es`.
    """

    n_dimensions: int
    initial_step_size: float
    n_offspring: int = 1
    d: Optional[float] = None
    p_target_succ: Optional[float] = None
    c_p: Optional[float] = None
    p_threshold: float = 0.44
    c_c: Optional[float] = None
    c_cov: Optional[float] = None

    def __post_init__(self) -> None:
        """Initialize variables with no provided value."""
        n_f = float(self.n_dimensions)
        n_offspring_f = float(self.n_offspring)
        if self.d is None:
            self.d = 1.0 + n_f / (2.0 * n_offspring_f)
        if self.p_target_succ is None:
            self.p_target_succ = 1.0 / (5.0 + math.sqrt(n_offspring_f) / 2.0)
        if self.c_p is None:
            tmp = self.p_target_succ * n_offspring_f
            self.c_p = tmp / (2.0 + tmp)
        if self.c_c is None:
            self.c_c = 2.0 / (n_f + 2.0)
        if self.c_cov is None:
            self.c_cov = 2.0 / (n_f * n_f + 6.0)


@dataclasses.dataclass
class OPLStoppingConditions(OptimizerStoppingConditions):
    """Define stopping conditions.

    Parameters
    ----------
    max_generations: optional
        Maximum number of generations
    max_evaluations: optional
        Maximum number of evaluations
    target_fitness_value: optional
        Target objective value.
    triggered: optional
        Indicates if any condition was triggered, when returned as an output.
    is_output: optional
        Indicates the class is used as an output.
    """

    max_generations: Optional[int] = None
    max_evaluations: Optional[int] = None
    target_fitness_value: Optional[float] = None
    triggered: bool = False
    is_output: dataclasses.InitVar[bool] = False

    def __post_init__(self, is_output):
        if (
            not is_output
            and self.max_generations is None
            and self.max_evaluations is None
            and self.target_fitness_value is None
        ):
            raise ValueError(
                "At least one stopping condition must be provided"
            )


@dataclasses.dataclass
class OPLSolution(OptimizerSolution):
    """A solution.

    Parameters
    ----------
    point
        The search point.
    fitness
        The objective point.
    """

    point: np.ndarray
    fitness: float


class OnePlusLambdaCMA(Optimizer):
    """The OnePlusLambda-CMA-ES optimizer.

    Parameters
    ----------
    parent_points:
        The search point of the initial parent.
    parent_fitness:
        The objective point of the initial parent.
    n_offspring
        Number of offspring.
    initial_step_size: optional
        Initial step size. Ignored if `parameters` is provided.
    max_generations: optional
        Maximum number of generations to trigger stop.
    max_evaluations: optional
        Maximum number of evaluations to trigger stop.
    target_fitness_value: optional
        Target fitness value to trigger stop.
    parameters: optional
        The external parameters. Allows to provide custom values other than \
        the recommended in the literature.
    rng: optional
        A random number generator.

    Notes
    -----
    Implements the algorithm defined in Algorithm 1, p. 4 :cite:`2007:mo-cma-es`.
    """

    _tell_points = True

    def __init__(
        self,
        parent_point: np.ndarray,
        parent_fitness: float,
        n_offspring: int,
        initial_step_size: float = 1e-4,
        max_generations: Optional[int] = None,
        max_evaluations: Optional[int] = None,
        target_fitness_value: Optional[float] = None,
        parameters: Optional[OPLParameters] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if not isinstance(parent_point, np.ndarray):
            raise ValueError("parent_point is not an array")
        self._n_dimensions = len(parent_point)

        if parameters is None:
            self._parameters = OPLParameters(
                n_dimensions=self._n_dimensions,
                n_offspring=n_offspring,
                initial_step_size=initial_step_size,
            )
        else:
            if parameters.n_dimensions != self._n_dimensions:
                raise ValueError(
                    "Invalid value for n_dimensions in parameters"
                )
            self._parameters = parameters

        self._stopping_conditions = OPLStoppingConditions(
            max_generations=max_generations,
            max_evaluations=max_evaluations,
            target_fitness_value=target_fitness_value,
        )

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

        self._point = parent_point
        self._fitness = parent_fitness
        self._penalized_fitness = parent_fitness
        self._step_size = self._parameters.initial_step_size
        self._p_succ = self._parameters.p_target_succ
        self._path = np.zeros(self._n_dimensions)
        self._cov = np.eye(self._n_dimensions)

        self._evaluation_count = 0
        self._generation_count = 0

    @property
    def name(self) -> str:
        return "OnePlusLambda-CMA-ES"

    @property
    def qualified_name(self) -> str:
        return f"(1+{self._parameters.n_offspring})-CMA-ES"

    @property
    def generation_count(self) -> int:
        """Return the number of elapsed generations."""
        return self._generation_count

    @property
    def evaluation_count(self) -> int:
        """Return the number of function evaluations."""
        return self._evaluation_count

    @property
    def parameters(self) -> OPLParameters:
        """Return a copy of the external parameters."""
        return dataclasses.replace(self._parameters)

    @property
    def stop(self) -> OPLStoppingConditions:
        conditions = self._stopping_conditions
        result = OPLStoppingConditions(is_output=True)

        if (
            conditions.max_generations is not None
            and self._generation_count >= conditions.max_generations
        ):
            result.triggered = True
            result.max_generations = self._generation_count
        if (
            conditions.max_evaluations is not None
            and self._evaluation_count >= conditions.max_evaluations
        ):
            result.triggered = True
            result.max_evaluations = self._evaluation_count
        if (
            conditions.target_fitness_value is not None
            and self._fitness < conditions.target_fitness_value
        ):
            result.triggered = True
            result.target_fitness_value = self._fitness
        return result

    @property
    def best(self) -> OPLSolution:
        return OPLSolution(np.copy(self._point), self._fitness)

    def ask(self):
        """Generate new search points.

        Returns
        -------
        np.ndarray
            A reference to the new search points.
        """
        return self._rng.multivariate_normal(
            self._point,
            (self._step_size * self._step_size) * self._cov,
            size=self._parameters.n_offspring,
        )

    def tell(
        self,
        points: np.ndarray,
        fitness: Union[np.ndarray, float],
        penalized_fitness: Optional[np.ndarray] = None,
        evaluation_count: Optional[int] = None,
    ) -> None:
        """
        Pass fitness information to the optimizer.

        Parameters
        ----------
        points
            The search points.
        fitness
            The objective points.
        penalized_fitness: optional
            The penalized fitness of the search points. \
            Use case: constrained functions.
        evaluation_count: optional
            Total evaluation count. Use case: noisy functions.
        """
        points = points.squeeze()
        if len(points.shape) < 2:
            points = points.reshape((1, len(points)))
            fitness = np.array([[fitness]])
            if penalized_fitness is not None:
                penalized_fitness = np.array([[fitness]])
        if penalized_fitness is None:
            penalized_fitness = fitness

        if evaluation_count is None:
            self._evaluation_count += len(points)
        else:
            self._evaluation_count = evaluation_count

        c_p = self._parameters.c_p
        c_c = self._parameters.c_c
        c_c_prod = c_c * (2.0 - c_c)
        c_c_sqrt = math.sqrt(c_c_prod)
        c_cov = self._parameters.c_cov
        d_inv = 1.0 / self._parameters.d
        p_target_succ = self._parameters.p_target_succ
        p_threshold = self._parameters.p_threshold
        if penalized_fitness is None:
            penalized_fitness = fitness
        p_succ = np.mean(
            np.logical_not(penalized_fitness > self._penalized_fitness)
        )

        old_step_size = self._step_size
        old_point = self._point.copy()

        self._p_succ = (1.0 - c_p) * self._p_succ + c_p * p_succ
        self._step_size = self._step_size * math.exp(
            d_inv * ((self._p_succ - p_target_succ) / (1.0 - p_target_succ))
        )

        best_idx = np.argsort(penalized_fitness)[0]
        if not penalized_fitness[best_idx] > self._penalized_fitness:
            self._point[:] = points[best_idx]
            self._penalized_fitness = penalized_fitness[best_idx]
            self._fitness = fitness[best_idx]
            self._path *= 1.0 - c_c
            if self._p_succ < p_threshold:
                x_step = (self._point - old_point) / old_step_size
                self._path += c_c_sqrt * x_step
                self._cov *= 1.0 - c_cov
                self._cov += c_cov * np.outer(self._path, self._path)
            else:
                self._cov = (1.0 - c_cov) * self._cov + c_cov * (
                    np.outer(self._path, self._path) + c_c_prod * self._cov
                )
        self._generation_count += 1


__all__ = [
    "OnePlusLambdaCMA",
    "OPLParameters",
    "OPLStoppingConditions",
    "OPLSolution",
]
