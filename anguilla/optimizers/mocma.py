"""This module contains implementations related to the MO-CMA-ES \
    algorithm for real-valued multi-objective optimization (MOO)."""
from __future__ import annotations

import enum
import dataclasses
import math
import numpy as np

from typing import List, Optional, Union, Callable

import anguilla.hypervolume as hv

from anguilla.dominance import fast_non_dominated_sort
from anguilla.fitness.base import AbstractObjectiveFunction


@dataclasses.dataclass
class MOParameters:
    """Parameters for MO-CMA-ES"""

    n_dimensions: int
    initial_step_size: float
    n_parents: int
    n_offspring: Optional[int] = None
    d: Optional[float] = None
    p_target_succ: Optional[float] = None
    c_p: Optional[float] = None
    p_threshold: float = 0.44
    c_c: Optional[float] = None
    c_cov: Optional[float] = None

    def __post_init__(self) -> None:
        """Initialize variables with no provided value."""
        n_f = float(self.n_dimensions)
        if self.n_offspring is None:
            self.n_offspring = self.n_parents
        if self.d is None:
            self.d = 1.0 + n_f / (2.0 * self.n_offspring)
        if self.p_target_succ is None:
            self.p_target_succ = 1.0 / (
                5.0 + math.sqrt(self.n_offspring * 2.0)
            )
        if self.c_p is None:
            tmp = self.p_target_succ * self.n_offspring
            self.c_p = tmp / (2.0 + tmp)
        if self.c_c is None:
            self.c_c = 2.0 / (n_f + 2.0)
        if self.c_cov is None:
            self.c_cov = 2.0 / (n_f * n_f + 6.0)


@dataclasses.dataclass
class MOStoppingConditions:
    """Define stopping criteria for MO-CMA-ES.

    Parameters
    ----------
    max_evaluation_count
        Maximum number of function evaluations.
    """

    max_evaluation_count: float = math.inf


class MOArchive:
    """A fixed-size population archive for MOCMA.

    Parameters
    ----------
    parameters
        The external parameters and meta-parameters.
    n_objectives
        The number of objectives.

    Attributes
    ----------
    points
        The search points of the individuals.
    offspring_parents
        The parents of each offspring.
    fitness
        The fitness values of the individuals.
    penalized_fitness
        The penalized fitness values of the individuals.
    fronts
        The fronts assigned by the non-dominated sorting of the individuals.
    contributions
        The contributions assigned by the quality indicator.
    selected
        A flag array indicating if an individual is selected.
    best
        The index of the individual with best quality indicator.
    p_succ
        The smoothed success probability of the individuals.
    step_size
        The step sizes of the individuals.
    cov_C
        The covariance matrices of the individuals.
    path_cov
        The evolution paths of the individuals.
    offspring_points
        A view for the search points of the offspring.
    offspring_fitness
        A view for the fitness values of the offspring.
    offspring_penalized_fitness
        A view for the penalized fitness values of the offspring.
    parent_points
        A view for the search points of the parents.
    parent_fitness
        A view for the fitness values of the parents.
    parent_penalized_fitness
        A view for the penalized fitness values of the parents.
    parent_fronts
        A view for the fronts assigned to parents.
    parent_p_succ
        A view for the smoothed success probability of the parents.
    parent_step_size
        A view for the step sizes of the parents.
    parent_cov_C
        A view for the covariance matrices of the parents.
    parent_path_cov
        A view for the path of the parents.
    """

    points: np.ndarray
    offspring_parents: np.ndarray
    fitness: np.ndarray
    fronts: np.ndarray
    contributions: np.ndarray
    selected: np.ndarray
    best: Optional[int]
    p_succ: np.ndarray
    step_size: np.ndarray
    cov_C: np.ndarray
    path_cov: np.ndarray
    # Convenience views
    offspring_points: np.ndarray
    offspring_fitness: np.ndarray
    offspring_penalized_fitness: np.ndarray
    parent_points: np.ndarray
    parent_fitness: np.ndarray
    parent_penalized_fitness: np.ndarray
    parent_fronts: np.ndarray
    parent_p_succ: np.ndarray
    parent_step_size: np.ndarray
    parent_cov_C: np.ndarray
    parent_path_cov: np.ndarray

    def __init__(self, parameters: MOParameters, n_objectives: int) -> None:
        """Initialize the archive."""
        n_dimensions = parameters.n_dimensions
        n_parents = parameters.n_parents
        n_offspring = parameters.n_offspring
        p_target_succ = parameters.p_target_succ
        initial_step_size = parameters.initial_step_size
        size = n_parents + n_offspring

        self.points = np.zeros((size, n_dimensions))
        self.offspring_parents = np.zeros(n_offspring, dtype=int)
        self.fitness = np.zeros((size, n_objectives))
        self.penalized_fitness = np.zeros((size, n_objectives))
        self.fronts = np.zeros((size,), dtype=int)
        self.contributions = np.zeros((size,))
        self.selected = np.zeros(size, dtype=bool)
        self.best = None
        self.p_succ = np.repeat(p_target_succ, size)
        self.step_size = np.repeat(initial_step_size, size)
        self.cov_C = np.zeros((size, n_dimensions, n_dimensions))
        for i in range(n_dimensions):
            self.cov_C[:, i, i] = 1.0
        self.path_cov = np.zeros((size, n_dimensions))

        # Convenience views
        self.offspring_points = self.points[n_parents:, :]
        self.offspring_fitness = self.fitness[n_parents:, :]
        self.offspring_penalized_fitness = self.penalized_fitness[
            n_parents:, :
        ]
        self.parent_points = self.points[:n_parents, :]
        self.parent_fitness = self.fitness[:n_parents, :]
        self.parent_penalized_fitness = self.penalized_fitness[:n_parents, :]
        self.parent_fronts = self.fronts[:n_parents]
        self.parent_p_succ = self.p_succ[:n_parents]
        self.parent_step_size = self.step_size[:n_parents]
        self.parent_cov_C = self.cov_C[:n_parents, :, :]
        self.parent_path_cov = self.path_cov[:n_parents, :]

    def select(self) -> None:
        """Copy data from selected individuals to parent memory."""
        self.parent_points[:] = self.points[self.selected]
        self.parent_fitness[:] = self.fitness[self.selected]
        self.parent_penalized_fitness[:] = self.penalized_fitness[
            self.selected
        ]
        self.parent_fronts[:] = self.fronts[self.selected]
        self.parent_p_succ[:] = self.p_succ[self.selected]
        self.parent_step_size[:] = self.step_size[self.selected]
        self.parent_cov_C[:, :, :] = self.cov_C[self.selected, :, :]
        self.parent_path_cov[:] = self.path_cov[self.selected]


ObjectiveFunction = Union[
    AbstractObjectiveFunction, Callable[[np.ndarray], np.ndarray]
]


class NotionOfSuccess(enum.Enum):
    """Define the notion of success of an offspring."""

    PopulationBased = 0
    ParentBased = 1


class MOCMA:
    """The MO-CMA-ES algorithm for real-valued multi-objective optimization."""

    _rng: np.random.Generator
    _ref_point: Optional[np.ndarray]
    _generations: int

    def __init__(
        self,
        parameters: MOParameters,
        stopping_conditions: Optional[MOStoppingConditions] = None,
        rng: np.random.Generator = None,
        notion_of_success: NotionOfSuccess = NotionOfSuccess.PopulationBased,
        ref_point: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the optimizer."""
        self.parameters = parameters
        if stopping_conditions is None:
            self.stopping_conditions = MOStoppingConditions()
        else:
            self.stopping_conditions = stopping_conditions
        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng
        self._notion_of_success = notion_of_success
        self._ref_point = ref_point
        self._evaluation_count = 0.0
        self._generations = 0

    @property
    def name(self) -> str:
        return "MO-CMA-ES"

    def create_archive(
        self,
        fn: Optional[AbstractObjectiveFunction] = None,
        parent_points: Optional[np.ndarray] = None,
        parent_fitness: Optional[np.ndarray] = None,
    ) -> MOArchive:
        """Create an archive for a fixed-size population.

        Parameters
        ----------
        fn: optional
            An objective function implementing \
                :class:`AbstractObjectiveFunction`.
        parent_points: optional
            The search points of the parents.
        parent_fitness: optional
            The fitness of the parents.

        Returns
        -------
        MOArchive
            The archive.

        Raises
        ------
        ValueError
            Invalid value provided for parameter(s).

        Notes
        -----
        Can be called as:

            * ``create_archive(fn)``
            * ``create_archive(fn, search_points)``
            * ``create_archive(search_points, fitness_values)``
        """
        n_parents = self.parameters.n_parents
        if fn is not None:
            archive = MOArchive(self.parameters, fn.n_objectives)
            if parent_points is None:
                archive.parent_points[:] = fn.random_points(n_parents)
            else:
                archive.parent_points[:] = parent_points
            (
                archive.parent_fitness[:],
                archive.parent_penalized_fitness[:],
            ) = fn.evaluate_with_penalty(archive.parent_points)
        else:
            if parent_fitness is None:
                raise ValueError("Invalid value for parent_fitness")
            if parent_points is None:
                raise ValueError("Invalid value for parent_points")
            archive = MOArchive(self.parameters, parent_points.shape[1])
            archive.parent_points[:] = parent_points
            archive.parent_fitness[:] = parent_fitness
            archive.parent_penalized_fitness[:] = parent_fitness

        fast_non_dominated_sort(
            archive.parent_penalized_fitness, out=archive.parent_fronts
        )

        if self._ref_point is None:
            self._ref_point = (
                np.max(archive.parent_penalized_fitness, axis=0) + 1e-4
            )

        return archive

    def ask(self, archive: MOArchive) -> None:
        """Produce new offspring by mutating the parents.

        Notes
        -----
        Implements lines 3-7 of algorithm 1 from :cite:`2010:mo-cma-es`.
        The offspring fitness value and the ranking of parents and offspring \
        must be computed afterwards with a desired strategy \
        (e.g. penalized / unpenalized evaluation, using hypervolume or other \
        indicator).
        """
        n_dimensions = self.parameters.n_dimensions
        n_parents = self.parameters.n_parents
        n_offspring = self.parameters.n_offspring

        shape = (n_offspring, n_dimensions)
        z = self._rng.standard_normal(size=shape)

        if n_parents == n_offspring:
            # Each parent produces one offspring
            for parent_idx in range(n_parents):
                offspring_idx = n_parents + parent_idx
                self._mutate(parent_idx, offspring_idx, z, archive)
        else:
            # Otherwise, non-dominated parents reproduce uniformly at random
            parent_candidates = np.argwhere(
                archive.parent_fronts == 1
            ).flatten()
            parent_indices = self._rng.choice(
                parent_candidates, n_offspring, replace=True
            )
            for i, parent_idx in enumerate(parent_indices):
                offspring_idx = n_parents + i
                self._mutate(parent_idx, offspring_idx, z, archive)

    def rank2(self, archive: MOArchive, evaluation_count: int) -> None:
        """Rank the new population (parents and their offspring).

        Parameters
        ----------
        archive
            The working archive.
        evaluation_count
            The updated evaluation count.

        Notes
        -----
        Performs two-level sorting as defined in :cite:`2007:mo-cma-es`. \
        First, non-dominated sorting and then sorting according to their \
        hypervolume contributions at each front level. \
            Here implemented as in :cite:`2008:shark`.
        """
        self._evaluation_count = float(evaluation_count)
        n_parents = self.parameters.n_parents
        n_offspring = self.parameters.n_offspring
        population_size = n_parents + n_offspring
        # Reset the selected flag array
        archive.selected[:] = False
        # Compute the hypervolume contributions
        archive.contributions[:] = hv.contributions(
            archive.parent_penalized_fitness, self._ref_point
        )
        # Sort by non-dominance relation
        fast_non_dominated_sort(
            archive.parent_penalized_fitness, out=archive.fronts
        )
        sorted_idx = np.argsort(archive.fronts)
        # Select individuals
        last_front_value = 1
        last_front_individuals = np.zeros_like(archive.selected, dtype=bool)
        k = 0
        while k < population_size:
            individual_idx = sorted_idx[k]
            if archive.fronts[individual_idx] > last_front_value:
                if k < n_parents:
                    last_front_value += 1
                    archive.selected |= last_front_individuals
                    last_front_individuals[:] = False
                else:
                    break
            last_front_individuals[individual_idx] = True
            k += 1
        # Select remaining from the last seen front using the
        # hypervolume contributions as second order criterion
        remaining = n_parents - np.sum(archive.selected)
        if remaining > 0:
            idx = np.argwhere(last_front_individuals).flatten()
            last_front_hvc = archive.contributions[idx]
            sorted_idx = idx[np.argsort(-last_front_hvc)]
            archive.selected[sorted_idx[:remaining]] = True

    def rank(self, archive: MOArchive, evaluation_count: int) -> np.ndarray:
        """Rank the new population (parents and their offspring).

        Parameters
        ----------
        archive
            The working archive.
        evaluation_count
            The updated evaluation count.

        Returns
        -------
            The index representing the selection.

        Notes
        -----
        Performs two-level sorting as defined in :cite:`2007:mo-cma-es`. \
        First, non-dominated sorting and then sorting according to their \
        hypervolume contributions at each front level. \
            Here implemented a bit different to :cite:`2008:shark`.
        """
        self._evaluation_count = float(evaluation_count)
        population_size = len(archive.fitness)
        n_parents = self.parameters.n_parents
        archive.contributions[:] = hv.contributions(
            archive.fitness, self._ref_point
        )
        # Here we perform the selection similar to in Shark's
        # "IndicatorBasedSelection.h":
        fast_non_dominated_sort(archive.penalized_fitness, out=archive.fronts)
        ranking_idx = np.argsort(archive.fronts)
        last_rank = 1
        selected_individuals_l: List[int] = []
        last_front_individuals_l: List[int] = []
        k = 0
        archive.best = None
        best_contribution = float("-inf")
        while k < population_size:
            individual = ranking_idx[k]
            if archive.fronts[individual] > last_rank:
                if k < n_parents:
                    last_rank += 1
                    selected_individuals_l += last_front_individuals_l
                    last_front_individuals_l = [individual]
                else:
                    break
            else:
                last_front_individuals_l.append(individual)

            if last_rank == 1 and archive.contributions[k] > best_contribution:
                best_contribution = archive.contributions[k]
                archive.best = k
            k += 1
        selected_individuals = np.array(selected_individuals_l, dtype=int)
        last_front_individuals = np.array(last_front_individuals_l, dtype=int)
        # Select best using HV indicator
        remaining = n_parents - len(selected_individuals)
        if remaining > 0:
            last_front_contrib = archive.contributions[last_front_individuals]
            tmp_idx = np.argsort(-last_front_contrib)
            tmp_idx = tmp_idx[:remaining]
            selected_individuals = np.concatenate(
                [selected_individuals, last_front_individuals[tmp_idx]],
            )
        archive.selected[:] = False
        archive.selected[selected_individuals] = True

    def tell(self, archive: MOArchive) -> None:
        """Perform adaptation and environmental selection."""
        n_offspring = self.parameters.n_offspring
        n_parents = self.parameters.n_parents
        c_p = self.parameters.c_p
        d_inv = 1.0 / self.parameters.d
        p_target_succ = self.parameters.p_target_succ
        p_target_succ_comp = 1.0 - p_target_succ
        p_threshold = self.parameters.p_threshold
        c_c = self.parameters.c_c
        c_c_scaler = c_c * (2.0 - c_c)
        c_c_scaler_sqrt = math.sqrt(c_c_scaler)
        c_cov = self.parameters.c_cov

        # Since we don't copy the parents parameters when creating the \
        # offspring, we first update all the offspring
        # and then all the parents (because some offspring may have the same
        # parent depending on the provided parameters at initialization)

        # Adapt offspring
        success_indicators = np.zeros(n_offspring)
        for i in range(n_offspring):
            oidx = n_parents + i
            pidx = archive.offspring_parents[i]
            success_indicator = self._success_indicator(pidx, oidx, archive)
            success_indicators[i] = success_indicator
            if archive.selected[oidx]:
                # Adapt p_succ
                p_succ = (1.0 - c_p) * archive.p_succ[
                    pidx
                ] + c_p * success_indicator
                archive.p_succ[oidx] = p_succ
                # Adapt step size
                archive.step_size[oidx] = archive.step_size[pidx] * math.exp(
                    d_inv * ((p_succ - p_target_succ) / p_target_succ_comp)
                )
                if archive.p_succ[oidx] < p_threshold:
                    z = (
                        archive.points[oidx] - archive.points[pidx]
                    ) / archive.step_size[pidx]
                    archive.path_cov[oidx] = (1.0 - c_c) * archive.path_cov[
                        pidx
                    ] + c_c_scaler_sqrt * z
                    archive.cov_C[oidx] = (1.0 - c_cov) * archive.cov_C[
                        pidx
                    ] + c_cov * np.outer(
                        archive.path_cov[oidx], archive.path_cov[oidx]
                    )
                else:
                    archive.path_cov[oidx] = (1.0 - c_c) * archive.path_cov[
                        pidx
                    ]
                    archive.cov_C[oidx] = (1.0 - c_cov) * archive.cov_C[
                        pidx
                    ] + c_cov * (
                        np.outer(
                            archive.path_cov[oidx], archive.path_cov[oidx]
                        )
                        + c_c_scaler * archive.cov_C[pidx]
                    )
        # Adapt parents
        for i in range(n_offspring):
            pidx = archive.offspring_parents[i]
            if archive.selected[pidx]:
                # update step size procedure, line 1, [2007:mo-cma-es]
                archive.p_succ[pidx] = (1.0 - c_p) * archive.p_succ[pidx]
                archive.p_succ[pidx] += c_p * success_indicators[i]
                # update step size procedure, line 2, [2007:mo-cma-es]
                tmp = (
                    d_inv
                    * (archive.p_succ[pidx] - p_target_succ)
                    / p_target_succ_comp
                )
                archive.step_size[pidx] *= math.exp(tmp)

        # Environmental selection
        archive.select()
        self._generations += 1

    def stop(self) -> bool:
        """Compute if any of the stopping conditions hold.

        Returns
        -------
        bool
            At least one stopping condition holds.
        """
        return (
            self._evaluation_count
            > self.stopping_conditions.max_evaluation_count
        )

    def _success_indicator(
        self, parent_idx: int, offspring_idx: int, archive: MOArchive
    ) -> float:
        if (
            self._notion_of_success == NotionOfSuccess.PopulationBased
            and archive.selected[offspring_idx]
        ) or (
            (archive.fronts[offspring_idx] < archive.fronts[parent_idx])
            or (
                archive.fronts[offspring_idx] == archive.fronts[parent_idx]
                and archive.contributions[offspring_idx]
                >= archive.contributions[parent_idx]
            )
        ):
            return 1.0
        return 0.0

    def _mutate(
        self,
        parent_idx: int,
        offspring_idx: int,
        z: np.ndarray,
        archive: MOArchive,
    ) -> None:
        """Perform mutation of a parent."""
        B, D = np.linalg.eigh(archive.cov_C[parent_idx])
        archive.points[offspring_idx] = (
            archive.points[parent_idx]
            + archive.step_size[parent_idx] * (B @ D) @ z[parent_idx]
        )
        n_parents = self.parameters.n_parents
        archive.offspring_parents[offspring_idx - n_parents] = parent_idx
