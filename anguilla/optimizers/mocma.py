"""This module contains implementations related to the MO-CMA-ES \
    algorithm for real-valued multi-objective optimization (MOO)."""
import enum
import dataclasses
import math
from typing import Any, Callable, Iterable, Optional

import numpy as np

from anguilla.optimizers.base import (
    Optimizer,
    OptimizerParameters,
    OptimizerSolution,
    OptimizerStoppingConditions,
    OptimizerResult,
)

from anguilla.dominance import fast_non_dominated_sort
from anguilla.selection import indicator_selection
from anguilla.indicators import Indicator, HypervolumeIndicator


@dataclasses.dataclass
class MOParameters(OptimizerParameters):
    """Parameters for MO-CMA-ES.

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
    :cite:`2007:mo-cma-es` and p. 489 :cite:`2010:mo-cma-es`.
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
class MOStoppingConditions(OptimizerStoppingConditions):
    """Define stopping criteria for MO-CMA-ES.

    Parameters
    ----------
    max_generations: optional
        Maximum number of generations.
    max_evaluations: optional
        Maximum number of function evaluations.
    target_indicator_value: optional
        Target indicator value.
    target_indicator_ref_point: optional
        Reference to compute the target indicator value.
    target_indicator_value_rtol: optional
        Relative tolerance for the target indicator value.
    triggered: optional
        Indicates if any condition was triggered, when returned as an output.

    Notes
    -----
    The class can be used as an input to specify stopping conditions and as an
    output to define the conditions that triggered the stop.
    """

    max_generations: Optional[int] = None
    max_evaluations: Optional[int] = None
    target_indicator_value: Optional[float] = None
    target_indicator_ref_point: Optional[np.ndarray] = None
    target_indicator_value_rtol: float = 1e-6
    triggered: bool = False


@dataclasses.dataclass
class MOSolution(OptimizerSolution):
    """Solution data for MO-CMA-ES.

    Parameters
    ----------
    points
        The current Pareto set approximation.
    fitness
        The current Pareto front approximation.
    """

    points: np.ndarray
    fitness: np.ndarray


class SuccessNotion(enum.IntEnum):
    """Define the notion of success of an offspring."""

    PopulationBased = 0
    IndividualBased = 1

    def __str__(self) -> str:
        if self.value == SuccessNotion.PopulationBased.value:
            return "P"
        return "I"


class FixedSizePopulation:
    """Models a fixed-size population of CMA-ES individuals.

    Parameters
    ----------
    n_dimensions
        Dimensionality of the search space.
    n_objectives
        Dimensionality of the objective space.
    n_parents
        Number of parents.
    n_offspring
        Number of offspring.

    Attributes
    ----------
    n_parents
        Number of parents.
    n_offspring
        Number of offspring.
    points
        Search points of the individuals.
    fitness
        Objective points of the individuals.
    penalized_fitness
        Penalized objective points of the individuals.
    step_size
        Step sizes of the individuals.
    p_succ
        Smoothed success probabilities of the individuals.
    path
        Evolution paths of the individuals.
    cov
        Covariance matrices of the individuals.
    parents
        Parent indices of the offspring.
    """

    def __init__(
        self,
        n_dimensions: int,
        n_objectives: int,
        n_parents: int,
        n_offspring: int,
    ) -> None:
        """Initialize the population."""
        n_individuals = n_parents + n_offspring
        self.n_parents = n_parents
        self.n_offspring = n_offspring
        self.points = np.zeros((n_individuals, n_dimensions))
        self.fitness = np.zeros((n_individuals, n_objectives))
        self.penalized_fitness = np.zeros((n_individuals, n_objectives))
        self.step_size = np.zeros((n_individuals,))
        self.p_succ = np.zeros((n_individuals,))
        self.path = np.zeros((n_individuals, n_dimensions))
        self.cov = np.zeros((n_individuals, n_dimensions, n_dimensions))
        for i in range(n_dimensions):
            self.cov[:, i, i] = 1.0
        self.parents = np.repeat(-1, n_individuals).astype(int)


class MOCMA(Optimizer):
    """The MO-CMA-ES multi-objective optimizer.

    Parameters
    ----------
    parent_points
        The search points of the initial population.
    parent_fitness
        The objective points of the initial population.
    n_offspring: optional
        The number of offspring. Defaults to the same number of parents.
    initial_step_size: optional
        The initial step size. Ignored if `parameters` is provided.
    success_notion: optional
        The notion of success.
    indicator: optional
        The indicator to use.
    stopping_conditions: optional
        The stopping conditions.
    parameters: optional
        The external parameters. Allows to provide custom values other than \
        the recommended in the literature.
    rng: optional
        A random number generator.

    Notes
    -----
    The implementation supports the algorithms presented in Algorithm 4, \
    p. 12, :cite:`2007:mo-cma-es` and Algorithm 1, p. 488 \
    :cite:`2010:mo-cma-es`.
    """

    def __init__(
        self,
        parent_points: np.ndarray,
        parent_fitness: np.ndarray,
        n_offspring: Optional[int] = None,
        initial_step_size: float = 1e-4,
        success_notion: SuccessNotion = SuccessNotion.PopulationBased,
        indicator: Optional[Indicator] = None,
        stopping_conditions: Optional[MOStoppingConditions] = None,
        parameters: Optional[MOParameters] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if len(parent_points.shape) < 2:
            parent_points = parent_points.reshape((1, len(parent_points)))
        if len(parent_fitness.shape) < 2:
            parent_fitness = parent_fitness.reshape((1, len(parent_fitness)))
        self._n_dimensions = parent_points.shape[1]
        self._n_objectives = parent_fitness.shape[1]
        self._success_notion = success_notion
        self._n_parents = parent_points.shape[0]
        if n_offspring is None:
            self._n_offspring = self._n_parents
        else:
            self._n_offspring = n_offspring

        if parameters is None:
            self._parameters = MOParameters(
                n_dimensions=self._n_dimensions,
                initial_step_size=initial_step_size,
            )
        else:
            self._parameters = parameters
            if self._parameters.n_dimensions != self._n_dimensions:
                raise ValueError(
                    "Invalid value for n_dimensions in provided parameters"
                )

        if stopping_conditions is None:
            self._stopping_conditions = MOStoppingConditions()
        else:
            self._stopping_conditions = stopping_conditions

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = np.random.default_rng()

        if indicator is None:
            self._indicator = HypervolumeIndicator()
        else:
            self._indicator = indicator

        self._population = FixedSizePopulation(
            n_dimensions=self._n_dimensions,
            n_objectives=self._n_objectives,
            n_parents=self._n_parents,
            n_offspring=self._n_offspring,
        )
        self._population.points[: self._n_parents, :] = parent_points
        self._population.fitness[: self._n_parents, :] = parent_fitness
        self._population.penalized_fitness[
            : self._n_parents, :
        ] = parent_fitness
        self._population.p_succ[:] = self._parameters.p_target_succ
        self._population.step_size[:] = self._parameters.initial_step_size

        self._evaluation_count = 0
        self._generation_count = 0

    @property
    def name(self) -> str:
        return "MO-CMA-ES"

    @property
    def qualified_name(self) -> str:
        return "({}+{})-MO-CMA-ES-{}".format(
            self._n_parents,
            self._n_offspring,
            self._success_notion,
        )

    @property
    def generation_count(self) -> int:
        """Return the number of elapsed generations."""
        return self._generation_count

    @property
    def evaluation_count(self) -> int:
        """Return the number of function evaluations."""
        return self._evaluation_count

    @property
    def parameters(self) -> MOParameters:
        """Return a copy of the external parameters."""
        return dataclasses.replace(self._parameters)

    @property
    def indicator(self) -> Indicator:
        """Return a reference of the indicator."""
        return self._indicator

    @property
    def best(self) -> MOSolution:
        return MOSolution(
            points=np.copy(self._population.points[: self._n_parents]),
            fitness=np.copy(self._population.fitness[: self._n_parents]),
        )

    @property
    def stop(self) -> MOStoppingConditions:
        conditions = self._stopping_conditions
        result = MOStoppingConditions()
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
        if conditions.target_indicator_value is not None:
            raise NotImplementedError()
        return result

    def ask(self) -> np.ndarray:
        # Compute offspring
        if self._n_offspring == self._n_parents:
            # Algorithm 1, line 4a [2010:mo-cma-es]
            oidx = self._n_parents
            for pidx in range(self._n_parents):
                self._mutate(pidx, oidx)
                oidx += 1
        else:
            # Algorithm 1, line 4b [2010:mo-cma-es]
            ranks, _ = fast_non_dominated_sort(
                self._population.penalized_fitness[: self._n_parents]
            )
            parents = np.argwhere(ranks == 1).flatten()
            chosen_parents = self._rng.choice(
                parents, size=self._n_offspring, replace=True
            )
            for oidx, pidx in zip(
                range(self._n_parents, self._n_parents + self._n_offspring),
                chosen_parents,
            ):
                self._mutate(pidx, oidx)
        return self._population.points[self._n_parents :]

    def tell(
        self,
        input_fitness: np.ndarray,
        input_penalized_fitness: Optional[np.ndarray] = None,
        evaluation_count: Optional[int] = None,
    ) -> None:
        # Convenience local variables
        n_parents = self._n_parents
        n_offspring = self._n_offspring
        points = self._population.points[:]
        fitness = self._population.fitness[:]
        penalized_fitness = self._population.penalized_fitness[:]
        p_succ = self._population.p_succ[:]
        step_size = self._population.step_size[:]
        path = self._population.path[:]
        cov = self._population.cov[:]
        parents = self._population.parents[:]

        # Update using input data
        fitness[n_parents:] = input_fitness
        if input_penalized_fitness is None:
            penalized_fitness[n_parents:] = input_fitness
        else:
            penalized_fitness[n_parents:] = input_penalized_fitness
        if evaluation_count is None:
            self._evaluation_count += len(input_fitness)
        else:
            self._evaluation_count = evaluation_count

        selected, ranks = indicator_selection(
            self._indicator, penalized_fitness, n_parents
        )

        # Perform adaptation
        # [2007:mo-cma-es] Algorithm 4, lines 7-10
        old_step_size = step_size.copy()
        for oidx in range(n_parents, n_parents + n_offspring):
            pidx = parents[oidx]
            success_indicator = self._success_indicator(
                oidx, pidx, selected, ranks
            )
            if selected[oidx]:
                self._update_step_size(oidx, success_indicator)
                x_step = (points[oidx] - points[pidx]) / old_step_size[pidx]
                self._update_covariance_matrix(oidx, x_step)
            if selected[pidx]:
                self._update_step_size(pidx, success_indicator)

        # Complete the selection process
        # [2007:mo-cma-es] Algorithm 4, lines 11-13
        points[:n_parents] = points[selected]
        fitness[:n_parents] = fitness[selected]
        penalized_fitness[:n_parents] = penalized_fitness[selected]
        step_size[:n_parents] = step_size[selected]
        cov[:n_parents] = cov[selected]
        path[:n_parents] = path[selected]
        p_succ[:n_parents] = p_succ[selected]

        self._generation_count += 1

    def fmin(
        self,
        fn: Callable,
        fn_args: Optional[Iterable[Any]] = None,
        fn_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> OptimizerResult:
        raise NotImplementedError()

    def _mutate(self, pidx: int, oidx: int) -> None:
        self._population.parents[oidx] = pidx
        # [2007:mo-cma-es] Algorithm 4, lines 3-6 (except evaluating fitness)
        # Sample point
        self._population.points[oidx, :] = self._rng.multivariate_normal(
            self._population.points[pidx],
            (
                self._population.step_size[pidx]
                * self._population.step_size[pidx]
                * self._population.cov[pidx]
            ),
        )
        # Copy parent data
        self._population.step_size[oidx] = self._population.step_size[pidx]
        self._population.cov[oidx, :, :] = self._population.cov[pidx, :, :]
        self._population.path[oidx, :] = self._population.path[pidx, :]
        self._population.p_succ[oidx] = self._population.p_succ[pidx]

    def _update_step_size(self, idx: int, p_succ: float) -> None:
        # [2007:mo-cma-es] Procedure in p. 4
        # [2010:mo-cma-es] Algorithm 1 in p. 2, lines 9-10, 17-18
        c_p = self._parameters.c_p
        d_inv = 1.0 / self._parameters.d
        p_target_succ = self._parameters.p_target_succ
        self._population.p_succ[idx] *= 1.0 - c_p
        self._population.p_succ[idx] += c_p * p_succ
        num = self._population.p_succ[idx] - p_target_succ
        den = 1.0 - p_target_succ
        self._population.step_size[idx] *= math.exp(d_inv * (num / den))

    def _update_covariance_matrix(self, idx: int, x_step: np.ndarray) -> None:
        # [2007:mo-cma-es] Procedure in p. 5
        # [2010:mo-cma-es] Algorithm in p. 2, lines 11-16
        c_c = self._parameters.c_c
        c_c_prod = c_c * (2.0 - c_c)
        c_c_sqrt = math.sqrt(c_c_prod)
        c_cov = self._parameters.c_cov
        self._population.path[idx] *= 1.0 - c_c
        if self._population.p_succ[idx] < self._parameters.p_threshold:
            self._population.path[idx] += c_c_sqrt * x_step

            self._population.cov[idx] *= 1.0 - c_cov
            self._population.cov[idx] += c_cov * np.outer(
                self._population.path[idx], self._population.path[idx]
            )
        else:
            path_prod = np.outer(
                self._population.path[idx], self._population.path[idx]
            )
            self._population.cov[idx] = (1.0 - c_cov) * self._population.cov[
                idx
            ] + c_cov * (path_prod + c_c_prod * self._population.cov[idx])

    def _success_indicator(
        self, oidx: int, pidx: int, selected: np.ndarray, ranks: np.ndarray
    ) -> float:
        success = False
        if self._success_notion == SuccessNotion.IndividualBased:
            # [2010:mo-cma-es] Section 3.1, p. 489
            success = ranks[oidx] <= ranks[pidx]
        else:
            # [2010:mo-cma-es] Section 3.2, p. 489
            success = selected[oidx]
        return float(success)
