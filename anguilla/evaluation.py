"""This module contains tools to conduct experiments for statistical \
evaluation."""
import os
import pathlib
import dataclasses
import multiprocessing
from functools import partial
from itertools import product
from typing import Iterable, Optional, List, Union

import numpy as np

from anguilla.fitness.base import ObjectiveFunction
from anguilla.optimizers.mocma import MOCMA
from anguilla.dominance import non_dominated_sort


@dataclasses.dataclass
class MOCMATrialParameters:
    """Define parameters for an independent trial of the MOCMA optimizer.

    Parameters
    ----------
    fn_cls
        The objective function class.
    fn_args: optional
        The positional arguments to instantiate `fn_cls` with.
    fn_kwargs: optional
        The keyword arguments to instantiate `fn_cls` with.
    fn_rng_seed: optional
        Seed to generate a specific RNG for the benchmark function.
    n_parents: optional
        The number of parents.
    initial_step_size: optional
        The initial step size.
    success_notion: optional
        The notion of success (`population` or `individual`).
    """

    fn_cls: ObjectiveFunction
    fn_args: Optional[Iterable] = None
    fn_kwargs: Optional[dict] = None
    fn_rng_seed: Optional[int] = None
    n_parents: int = 20
    initial_step_size: float = 1.0
    success_notion: str = "population"
    n_offspring: Optional[int] = None
    reference: Optional[np.ndarray] = None
    region_bounds: Optional[np.ndarray] = None
    max_generations: Optional[int] = None
    max_evaluations: Optional[int] = None
    target_indicator_value: Optional[float] = None
    # Override with dataclasses.replace
    key: Optional[int] = None
    seed: Optional[int] = None

    def __post_init__(self):
        if self.n_offspring is None:
            self.n_offspring = self.n_parents
        if self.fn_args is None:
            self.fn_args = ()
        if self.fn_kwargs is None:
            self.fn_kwargs = {}


@dataclasses.dataclass
class LogParameters:
    """Define parameters for a trial log.

    Parameters
    ----------
    path
        Base path to save the log file.
    """

    path: Union[str, pathlib.Path]
    log_at: List[int]

    def __post_init__(self):
        if not isinstance(self.path, pathlib.Path):
            self.path = pathlib.Path(self.path)


def log_mocma_trial(
    log_parameters: LogParameters, trial_parameters: MOCMATrialParameters
) -> None:
    """Run an independent trial of the optimizer and log to a CSV file.

    Parameters
    ----------
    log_parameters
        The paramters to configure the logging.
    trial_parameters
        The parameters to configure the trial run.
    """
    if trial_parameters.seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(trial_parameters.seed)

    if trial_parameters.fn_rng_seed is not None:
        trial_parameters.fn_kwargs["rng"] = np.random.default_rng(
            trial_parameters.fn_rng_seed
        )
    else:
        trial_parameters.fn_kwargs["rng"] = rng

    fn = trial_parameters.fn_cls(
        *trial_parameters.fn_args, **trial_parameters.fn_kwargs
    )

    max_evaluations = max(*log_parameters.log_at)
    if trial_parameters.max_evaluations is not None:
        max_evaluations = max(
            trial_parameters.max_evaluations,
            max_evaluations,
        )

    parent_points = fn.random_points(
        trial_parameters.n_parents,
        region_bounds=trial_parameters.region_bounds,
    )
    parent_fitness = fn(parent_points)

    optimizer = MOCMA(
        parent_points,
        parent_fitness,
        n_offspring=trial_parameters.n_offspring,
        rng=rng,
        success_notion=trial_parameters.success_notion,
        max_generations=trial_parameters.max_generations,
        max_evaluations=max_evaluations,
        target_indicator_value=trial_parameters.target_indicator_value,
    )
    if trial_parameters.reference is not None:
        optimizer.indicator.reference = trial_parameters.reference
    while not optimizer.stop.triggered:
        points = optimizer.ask()
        if fn.has_constraints:
            fitness = fn.evaluate_with_penalty(points)
            optimizer.tell(*fitness)
        else:
            fitness = fn(points)
            optimizer.tell(fitness)
        if optimizer.evaluation_count in log_parameters.log_at:
            fname = log_parameters.path.joinpath(
                "{}_{}_{}_{}.csv".format(
                    optimizer.qualified_name,
                    fn.name,
                    trial_parameters.key,
                    optimizer.evaluation_count,
                )
            )
            np.savetxt(
                str(fname.absolute()),
                optimizer.best.fitness,
                delimiter=",",
            )


def log_mocma_trials(
    log_parameters: LogParameters,
    trial_parameters: List[MOCMATrialParameters],
    trial_slice: Optional[slice] = None,
    seed: int = 0,
    n_trials: int = 5,
    n_processes: int = 5,
) -> None:
    """Run independent trials of MOCMA with a benchmark function and save \
    fitness data to CSV files.

    Parameters
    ----------
    log_parameters
        Log parameters.
    trial_parameters
        Common parameters for the trials.
    seed: optional
        Base seed for the group of trials.
    n_processes: optional
        Number of CPUs to use.

    Notes
    -----
    Generates seeds for each trial using `SeedSequence` from Numpy.
    See ` this page for more information. <https://bit.ly/360PTDN>`_.
    """
    # Create seeds for independent trials
    # See: https://numpy.org/doc/stable/reference/random/parallel.html
    seed_sequence = np.random.SeedSequence(seed)
    seeds = seed_sequence.spawn(n_trials * len(trial_parameters))

    seeded_trial_parameters = list(
        zip(product(trial_parameters, range(n_trials)), seeds)
    )

    if trial_slice is None:
        trial_slice = slice(len(seeds))

    params = list(
        dataclasses.replace(data, seed=seed, key=trial_id + 1)
        for (data, trial_id), seed in seeded_trial_parameters[trial_slice]
    )

    n_processes = min(n_processes, len(params))
    cpu_count = os.cpu_count()
    chunksize = 1
    if n_processes > cpu_count:
        print(
            """Provided number of processes ({}) """
            """exceeds available CPUs ({}).""".format(n_processes, cpu_count)
        )
        chunksize = (
            n_processes // cpu_count + 1 if n_processes % cpu_count != 0 else 0
        )
        n_processes = cpu_count

    if n_processes > 1:
        print(
            "Running {} trials using {} processes with chunk size of {}.".format(
                n_trials,
                n_processes,
                chunksize,
            )
        )
        with multiprocessing.Pool(processes=n_processes) as pool:
            pool.map(
                partial(log_mocma_trial, log_parameters),
                params,
                chunksize=chunksize,
            )
    else:
        print(
            "Running {} trial(s) using sequential execution".format(n_trials)
        )
        map(partial(log_mocma_trial, log_parameters), params)


def union_upper_bound(
    *populations: Iterable[np.ndarray], translate_by: float = 0.0
):
    """Compute the upper bound of the union of non-dominated individuals \
    in a set of populations.

    Parameters
    ----------
    populations
        The populations of individuals to build the reference set with.
    translate_by: optional
        Number to translate the upper bound with.

    Returns
    -------
        The translated upper bound of the union of non-nominated individuals.

    Notes
    -----
    Based on the reference set described in Section 4.2, p. 490 \
    :cite:`2010:mo-cma-es`.
    """
    reference_set = np.vstack(populations)
    ranks, _ = non_dominated_sort(reference_set, 1)
    reference_set = reference_set[ranks == 1]
    return np.max(reference_set, axis=0) + translate_by


__all__ = [
    "log_mocma_trials",
    "union_upper_bound",
]