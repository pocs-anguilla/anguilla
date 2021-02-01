"""This module contains implementations related to the UP-MO-CMA-ES \
    algorithm for real-valued multi-objective optimization (MOO)."""
import dataclasses
import enum
import math
from typing import Any, Iterable, Optional

import numpy as np

from anguilla.optimizers.base import (
    Optimizer,
    OptimizerStoppingConditions,
    OptimizableFunction,
    OptimizerResult,
)

from anguilla.dominance import dominates

from anguilla.archive import UPMOArchive, UPMOParameters


@dataclasses.dataclass
class UPMOStoppingConditions(OptimizerStoppingConditions):
    """Define stopping criteria for UP-MO-CMA-ES.

    Parameters
    ----------
    max_generations: optional
        Maximum number of generations.
    max_evaluations: optional
        Maximum number of function evaluations.
    max_nbytes: optional
        Maximum memory the archive can utilize (in bytes).
    max_size: optional
        Maximum number of individuals.
    triggered: optional
        Indicates if any condition was triggered, when returned as an output.
    is_output: optional
        Indicates the class is used as an output.
    Notes
    -----
    The class can be used as an input to specify stopping conditions and as an
    output to define the conditions that triggered the stop.
    """

    max_generations: Optional[int] = None
    max_evaluations: Optional[int] = None
    max_nbytes: Optional[int] = None
    max_size: Optional[int] = None
    triggered: bool = False
    is_output: dataclasses.InitVar[bool] = False

    def __post_init__(self, is_output):
        if (
            not is_output
            and self.max_generations is None
            and self.max_evaluations is None
            and self.max_nbytes is None
            and self.max_size is None
        ):
            raise ValueError(
                "At least one stopping condition must be provided"
            )


class SuccessNotion(enum.IntEnum):
    """Define the notion of success of an offspring."""

    PopulationBased = 0
    IndividualBased = 1

    def __str__(self) -> str:
        if self.value == SuccessNotion.PopulationBased.value:
            return "P"
        return "I"


class CovModel(enum.IntEnum):
    """Define how to store the covariance information."""

    Full = 0
    Cholesky = 1

    def __str__(self) -> str:
        if self.value == CovModel.Full.value:
            return "F"
        return "C"


class UPMOCMA(Optimizer):
    """The UP-MO-CMA-ES optimizer.

    Parameters
    ----------
    initial_points
        The search points of the initial population.
    initial_fitness
        The objective points of the initial population.
    initial_step_size: optional
        The initial step size. Ignored if `parameters` is provided.
    success_notion: optional
        The notion of success (either `individual` or `population`).
    reference: optional
        A reference point.
    parameters: optional
        The external parameters. Allows to provide custom values other than \
        the recommended in the literature.
    rng: optional
        A random number generator.
    cov_model: optional
        How to store the covariance information (either `full` or `cholesky`).

    Raises
    ------
    ValueError
        A parameter was provided with an invalid value.
    NotImplementedError
        A parameter value is not supported yet.

    Notes
    -----
    The implementation is based on :cite:`2016:mo-cma-es`.
    """

    def __init__(
        self,
        initial_points: np.ndarray,
        initial_fitness: np.ndarray,
        initial_step_size: float = 1.0,
        max_generations: Optional[int] = None,
        max_evaluations: Optional[int] = None,
        max_nbytes: Optional[int] = None,
        max_size: Optional[int] = None,
        success_notion: str = "population",
        reference: Optional[np.ndarray] = None,
        parameters: Optional[UPMOParameters] = None,
        rng: Optional[np.random.Generator] = None,
        cov_model: str = "full",
    ):
        self._n_dimensions = initial_points.shape[1]
        self._n_objectives = initial_fitness.shape[1]
        if self._n_objectives != 2:
            raise NotImplementedError("Unsupported objective dimensionality")

        if success_notion == "individual":
            self._success_notion = SuccessNotion.IndividualBased
        elif success_notion == "population":
            self._success_notion = SuccessNotion.PopulationBased
        else:
            raise ValueError("Invalid value for success_notion.")

        if cov_model == "full":
            self._cov_model = CovModel.Full
        elif cov_model == "cholesky":
            raise NotImplementedError(
                "Support for Cholesky factors has not been implemented."
            )
        else:
            raise ValueError("Invalid value for cov_model.")

        if parameters is None:
            self._parameters = UPMOParameters(
                self._n_dimensions,
                initial_step_size,
            )
        else:
            self._parameters = parameters
            if self._parameters.n_dimensions != self._n_dimensions:
                raise ValueError(
                    "Invalid value for n_dimensions in provided parameters"
                )

        self._stopping_conditions = UPMOStoppingConditions(
            max_generations=max_generations,
            max_evaluations=max_evaluations,
            max_nbytes=max_nbytes,
            max_size=max_size,
        )

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

        self._population = UPMOArchive(self._parameters, reference)
        for point, fitness in zip(initial_points, initial_fitness):
            self._population.insert(point, fitness)

        self._generation_count = 0
        self._evaluation_count = 0
        self._parent = None
        self._offspring_cov = None
        self._offspring_point = None
        self._ask_called = False

    @property
    def name(self):
        """Return the name of the optimizer.

        Returns
        -------
        The optimizer's name.
        """
        return "UP-MO-CMA-ES-{}".format(str(self._success_notion))

    @property
    def qualified_name(self):
        """Return the qualified name of the optimizer.

        Returns
        -------
        The optimizer's qualified name.
        """
        return self.name

    @property
    def generation_count(self) -> int:
        """Return the number of elapsed generations."""
        return self._generation_count

    @property
    def evaluation_count(self) -> int:
        """Return the number of function evaluations."""
        return self._evaluation_count

    @property
    def parameters(self) -> UPMOParameters:
        """Return a read-only version of the external parameters."""
        return self._parameters

    @property
    def size(self) -> int:
        """Return the size of the population.

        Returns
        -------
        The size of the population.
        """
        return self._population.size

    @property
    def nbytes(self) -> int:
        """Return the memory in bytes used by the population archive."""
        return self._population.nbytes

    def ask(self) -> Any:
        """Generate a new search point.

        Returns
        -------
        np.ndarray
            The new search point.

        Raises
        ------
        RuntimeError
            When called before initializing the population with \
            `insert_initial`.
        """
        # Information we need to persist between calls to ask and tell:
        # * Reference to parent
        # * Offspring covariance matrix
        # * Offspring search point
        sigma_min = self._parameters.sigma_min
        p_extreme = self._parameters.p_extreme
        c_r = self._parameters.c_r
        c_r_h = 0.5 * c_r
        p = self._rng.uniform(size=2)
        if p[0] < p_extreme or self._population.size <= 2:
            self._parent = self._population.sample_extreme(p[1])
            if self._parent.step_size < sigma_min:
                self._parent = self._population.sample_interior(p[1])
        else:
            self._parent = self._population.sample_interior(p[1])
        nearest = self._population.nearest(self._parent)
        self._offspring_cov = (1.0 - c_r) * self._parent.cov
        if nearest[0] is not None:
            z = (
                nearest[0].point - self._parent.point
            ) / self._parent.step_size
            self._offspring_cov += c_r_h * np.outer(z, z)
        if nearest[1] is not None:
            z = (
                nearest[1].point - self._parent.point
            ) / self._parent.step_size
            self._offspring_cov += c_r_h * np.outer(z, z)

        self._offspring_point = self._rng.multivariate_normal(
            self._parent.point,
            (self._parent.step_size * self._parent.step_size)
            * self._offspring_cov,
        )

        self._ask_called = True
        return self._offspring_point

    def tell(self, fitness: np.ndarray, evaluation_count: int = 1) -> None:
        """
        Pass fitness information to the optimizer.

        Parameters
        ----------
        fitness
            The fitness of the search point.
        evaluation_count: optional
            Total evaluation count. Use case: noisy functions.

        Raises
        ------
        RuntimeError
            When `tell` is called before `ask`.

        Notes
        -----
        Assumes stored offspring data (i.e covariance matrix) corresponds to
        the search point produced by the last call to `ask`.
        """
        if not self._ask_called:
            raise RuntimeError("Tell called before ask")
        z = (
            self._offspring_point - self._parent.point
        ) / self._parent.step_size
        # If the offspring dominates the parent, the last will be
        # deleted when inserting the first
        if dominates(fitness, self._parent.fitness):
            self._parent = None
        # We attempt to insert the point into the archive
        offspring = self._population.insert(self._offspring_point, fitness)
        # If the offspring was not inserted it is because it was dominated
        # and hence unsuccessful.
        if offspring is not None:
            # With population-based notion of success if it is
            # inserted it is successful.
            success_indicator = 1.0
            # With individual-based notion of success if its contribution
            # is greater or equal to its parent's contribution it is successful
            if (
                self._success_notion == SuccessNotion.IndividualBased
                and offspring.contribution < self._parent.contribution
            ):
                success_indicator = 0.0
            c_p = self._parameters.c_p
            c_cov = self._parameters.c_cov
            d_inv = 1.0 / self._parameters.d
            p_target_succ = self._parameters.p_target_succ
            p_target_succ_comp = 1.0 - p_target_succ
            zz = np.outer(z, z)
            # Adapt offspring
            offspring.p_succ *= 1.0 - c_p
            offspring.p_succ += c_p * success_indicator
            offspring.step_size *= math.exp(
                d_inv
                * ((offspring.p_succ - p_target_succ) / p_target_succ_comp)
            )
            offspring.cov[:, :] = (
                1.0 - c_cov
            ) * self._offspring_cov + c_cov * zz
            # Adapt parent if it was not deleted
            if self._parent is not None:
                self._parent.p_succ *= 1.0 - c_p
                self._parent.p_succ += c_p * success_indicator
                self._parent.step_size *= math.exp(
                    d_inv
                    * (
                        (self._parent.p_succ - p_target_succ)
                        / p_target_succ_comp
                    )
                )
                self._parent.cov[:, :] = (
                    1.0 - c_cov
                ) * self._parent.cov + c_cov * zz
        self._offspring_cov = None
        self._offspring_point = None
        self._generation_count += 1
        self._evaluation_count += evaluation_count
        self._ask_called = False

    @property
    def stop(self) -> UPMOStoppingConditions:
        conditions = self._stopping_conditions

        result = UPMOStoppingConditions(is_output=True)
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
            conditions.max_size is not None
            and self._population.size >= conditions.max_size
        ):
            result.triggered = True
            result.max_size = self._population.size
        if (
            conditions.max_nbytes is not None
            and self._population.max_nbytes >= conditions.max_nbytes
        ):
            result.triggered = True
            result.max_nbytes = self._population.max_nbytes
        return result

    def best(self):
        raise NotImplementedError()

    def fmin(
        self,
        fn: OptimizableFunction,
        fn_args: Optional[Iterable[Any]],
        fn_kwargs: Optional[dict],
        **kwargs: Any,
    ) -> OptimizerResult:
        raise NotImplementedError()


__all__ = [
    "UPMOCMA",
    "UPMOArchive",
    "UPMOParameters",
]
