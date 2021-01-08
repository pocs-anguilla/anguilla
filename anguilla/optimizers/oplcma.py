"""This module contains implementations related to the OnePlusLambda-CMA-ES \
    algorithm for real-valued single-objective optimization (MOO)."""
import dataclasses
import math
from typing import Any, Callable, Iterable, Optional

import numpy as np

from anguilla.fitness.base import ObjectiveFunction
from anguilla.optimizers.base import (
    Optimizer,
    OptimizerParameters,
    OptimizerSolution,
    OptimizerStoppingConditions,
    OptimizerResult,
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


@dataclasses.dataclass
class OPLStoppingConditions(OptimizerStoppingConditions):
    """Define stopping conditions.

    Parameters
    ----------
    max_generations: optional
        Maximum number of generations
    max_evaluations: optional
        Maximum number of evaluations
    target_fitness: optional
        Target objective value.
    target_fitness_rtol: optional
        Relative tolerance of target objective value.
    triggered: optional
    """

    max_generations: Optional[int] = None
    max_evaluations: Optional[int] = None
    target_fitness: Optional[float] = None
    target_fitness_rtol: float = 1e-6
    triggered: bool = False


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
        Initial step size.
    stopping_conditions: optional
        The stopping conditions.
    parameters: optional
        The external parameters. Allows to provide custom values other than \
        the recommended in the literature.
    rng: optional
        A random number generator.

    Notes
    -----
    Implements the algorithm defined in Algorithm 1, p. 4 :cite:`2007:mo-cma-es`.
    """

    def __init__(
        self,
        parent_point: np.ndarray,
        parent_fitness: np.ndarray,
        n_offspring: int,
        initial_step_size: float = 1e-4,
        parameters: Optional[OPLParameters] = None,
        stopping_conditions: Optional[OPLStoppingConditions] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
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

        if stopping_conditions is None:
            self._stopping_conditions = OPLStoppingConditions()
        else:
            self._stopping_conditions = stopping_conditions

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

        self.point = parent_point
        self.fitness = parent_fitness
        self.penalized_fitness = parent_fitness
        self.step_size = self._parameters.initial_step_size
        self.p_succ = self._parameters.p_target_succ
        self.path = np.zeros(n_dimensions)
        self.cov = np.eye(n_dimensions)

        self._evaluation_count = 0
        self._generation_count = 0

    @property
    def name(self) -> str:
        return "OnePlusLambda-CMA-ES"

    @property
    def qualified_name(self) -> str:
        return f"(1+{self._parameters.n_offspring})-CMA-ES"

    @property
    def generation_count() -> int:
        """The number of elapsed generations."""
        return self._generation_count

    @property
    def evaluation_count() -> int:
        """The number of function evaluations."""
        return self._evaluation_count

    @property
    def parameters(self) -> MOParameters:
        """Return a copy of the external parameters."""
        return dataclasses.replace(self._parameters)

    @property
    def stop(self) -> OPLStoppingConditions:
        conditions = self._stopping_conditions
        result = OPLStoppingConditions()
        max_generations = self._stopping_conditions.max_generations
        max_evaluations = self._stopping_conditions.max_evaluations
        target_fitness = self._stopping_conditions.target_fitness

        if (
            conditions.max_generations is not None
            and self._generation_count < max_generations
        ):
            result.triggered = True
            result.max_generations = self._generation_count
        if max_evaluations is None or self._evaluation_count < max_evaluations:
            result.triggered = True
            result.max_evaluations = self._evaluation_count
        if conditions.target_fitness is None and math.isclose(
            conditions.target_fitness,
            self.fitness,
            rel_tol=conditions.target_fitness_rtol,
        ):
            result.triggered = True
            result.target_fitness = self.fitness
        return result

    @property
    def best(self) -> OPLSolution:
        return OPLSolution(np.copy(self.point), self.fitness)

    def ask(self):
        return self._rng.multivariate_normal(
            self.point,
            (self.step_size * self.step_size) * self.cov,
            size=self._parameters.n_offspring,
        )

    def tell(
        self,
        points: np.ndarray,
        fitness: np.ndarray,
        penalized_fitness: Optional[np.ndarray] = None,
        evaluation_count: Optional[int] = None,
    ) -> None:
        c_p = self._parameters.c_p
        c_c = self._parameters.c_c
        c_c_prod = c_c * (2.0 - c_c)
        c_c_sqrt = math.sqrt(c_c_prod)
        c_cov = self._parameters.c_cov
        d_inv = 1.0 / self._parameters.d
        p_target_succ = self._parameters.p_target_succ
        p_threshold = self._parameters.p_threshold
        p_succ = np.mean(
            np.logical_not(penalized_fitness > self.penalized_fitness)
        )

        old_step_size = self.step_size
        old_point = self.point.copy()

        self.p_succ = (1.0 - c_p) * self.p_succ + c_p * p_succ
        self.step_size = self.step_size * math.exp(
            d_inv * ((self.p_succ - p_target_succ) / (1.0 - p_target_succ))
        )

        if len(points.shape) < 2:
            points = points.reshape((1,))
            fitness = fitness.reshape((1,))
            if penalized_fitness is not None:
                penalized_fitness = penalized_fitness.reshape((1,))
        if penalized_fitness is None:
            penalized_fitness = fitness

        if evaluation_count is None:
            self._evaluation_count += len(parent_points)
        else:
            self._evaluation_count = evaluation_count

        best_idx = np.argsort(penalized_fitness)[0]
        if not (penalized_fitness[best_idx] > self.penalized_fitness):
            self.point[:] = points[best_idx]
            self.penalized_fitness = penalized_fitness[best_idx]
            self.fitness = fitness[best_idx]
            self.path *= 1.0 - c_c
            if self.p_succ < p_threshold:
                x_step = (self.point - old_point) / old_step_size
                self.path += c_c_sqrt * x_step
                self.cov *= 1.0 - c_cov
                self.cov += c_cov * np.outer(self.path, self.path)
            else:
                self.cov = (1.0 - c_cov) * self.cov + c_cov * (
                    np.outer(self.path, self.path) + c_c_prod * self.cov
                )
        self._generation_count += 1

    def fmin(
        self,
        fn: Callable,
        fn_args: Optional[Iterable[Any]] = (),
        fn_kwargs: Optional[dict] = {},
        **kwargs: Any,
    ) -> OptimizerResult:
        raise NotImplementedError()
