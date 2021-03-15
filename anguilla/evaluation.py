"""This module contains tools to conduct experiments for statistical \
evaluation."""
import abc
import os
import platform
import re
import time
import pathlib
import dataclasses
import multiprocessing
from functools import partial
from itertools import product
from typing import Iterable, Optional, List, Union

import numpy as np

import anguilla
from anguilla.fitness.base import ObjectiveFunction

# from anguilla.optimizers.mocma import MOCMA
from anguilla.optimizers import MOCMA


class StopWatch:
    """Keep wall-clock time between operations."""

    def __init__(self):
        self._started_at = None
        self._duration = 0

    def start(self):
        """Start the clock."""
        self._started_at = time.perf_counter()

    def stop(self):
        """Stop the clock.

        Raises
        ------
        ValueError
            Stop was called before start.
        """
        if self._started_at is None:
            raise ValueError("Stop called before start")
        self._duration += time.perf_counter() - self._started_at
        self._started_at = None

    @property
    def duration(self):
        """Cumulative elapsed wall-clock time in seconds between stops."""
        return self._duration


class TrialParameters(metaclass=abc.ABCMeta):
    """Abstract trial parameters."""


@dataclasses.dataclass
class MOCMATrialParameters(TrialParameters):
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
    n_offspring: optional
        The number of offspring per parent.
    reference: optional
        A reference point for the hypervolume operations.
    region_bounds: optional
        Region bounds for the initial search points.
    max_generations: optional
        Maximum number of generations to trigger stop.
    max_evaluations: optional
        Maximum number of function evaluations to trigger stop.
    target_indicator_value: optional
        Target value of the indicator to trigger stop.
    key: optional
        The trial identifier.
    seed: optional
        The trial random seed. Does not affect the objective function if \
        `fn_rng_seed` is provided.
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
class UPMOCMATrialParameters(TrialParameters):
    """Define parameters for an independent trial of the UPMOCMA optimizer.

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
    n_starting_points: optional
        The number of starting points to generate.
    initial_step_size: optional
        The initial step size.
    success_notion: optional
        The notion of success (`population` or `individual`).
    reference: optional
        A reference point for the hypervolume operations.
    region_bounds: optional
        Region bounds for the initial search points.
    max_generations: optional
        Maximum number of generations to trigger stop.
    max_evaluations: optional
        Maximum number of function evaluations to trigger stop.
    max_nbytes: optional
        Maximum memory the archive can utilize (in bytes).
    max_size: optional
        Maximum number of individuals.
    key: optional
        The trial identifier.
    seed: optional
        The trial random seed. Does not affect the objective function if \
        `fn_rng_seed` is provided.
    """

    fn_cls: ObjectiveFunction
    fn_args: Optional[Iterable] = None
    fn_kwargs: Optional[dict] = None
    fn_rng_seed: Optional[int] = None
    n_starting_points: Optional[int] = None
    initial_step_size: float = 1.0
    success_notion: str = "population"
    reference: Optional[np.ndarray] = None
    region_bounds: Optional[np.ndarray] = None
    max_generations: Optional[int] = None
    max_evaluations: Optional[int] = None
    max_nbytes: Optional[int] = None
    max_size: Optional[int] = None
    # Override with dataclasses.replace
    key: Optional[int] = None
    seed: Optional[int] = None

    def __post_init__(self):
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
    log_at
        Evaluation counts at which to log.
    log_fitness: optional
        True if objective points should be logged.
    log_points: optional
        True if search points should be logged.
    log_step_sizes: optional
        True if step sizes should be logged.
    cpu_info: optional
        String with CPU information metadata.
    """

    path: Union[str, pathlib.Path]
    log_at: List[int]
    log_fitness: bool = True
    log_points: bool = False
    log_step_sizes: bool = False
    cpu_info: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.path, pathlib.Path):
            self.path = pathlib.Path(self.path)


def log_mocma_trial(
    log_parameters: LogParameters, trial_parameters: MOCMATrialParameters
) -> str:
    """Run an independent trial of the optimizer and log to a CSV file.

    Parameters
    ----------
    log_parameters
        The paramters to configure the logging.
    trial_parameters
        The parameters to configure the trial run.

    Returns
    -------
    str
        A string identifying the job.
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
        seed=trial_parameters.seed.generate_state(1)[0],
        # rng=rng,
        success_notion=trial_parameters.success_notion,
        max_generations=trial_parameters.max_generations,
        max_evaluations=max_evaluations,
        target_indicator_value=trial_parameters.target_indicator_value,
    )
    if trial_parameters.reference is not None:
        optimizer.indicator.reference = trial_parameters.reference

    cpu_info = log_parameters.cpu_info
    uname = platform.uname()
    os_info = "{} {}".format(uname.system, uname.release)
    machine_info = uname.machine
    if cpu_info is not None:
        machine_info = cpu_info
    python_info = "{}.{}.{}".format(*platform.python_version_tuple())

    header = (
        """Generated with {} {}, {} {}\n"""
        """Machine: {}\n"""
        """OS: {}\n"""
        """Python: {}\n"""
        """Optimizer: {}\n"""
        """Function: {}: {} -> {}\n"""
        """Initial step size: {}\n"""
        """Reference point: {}\n"""
        """Trial seed: entropy={}, spawn_key={}\n"""
        """Function-specific seed: {}\n"""
        """Trial: {}\n"""
        """Evaluations: {{}}\n"""
        """Elapsed time (wall-clock): {{:.2f}}s\n"""
        """Observation: {{}}\n""".format(
            anguilla.__name__,
            anguilla.__version__,
            np.__name__,
            np.__version__,
            machine_info,
            os_info,
            python_info,
            optimizer.qualified_name,
            fn.qualified_name,
            fn.n_dimensions,
            fn.n_objectives,
            trial_parameters.initial_step_size,
            trial_parameters.reference,
            trial_parameters.seed.entropy,
            trial_parameters.seed.spawn_key,
            trial_parameters.fn_rng_seed,
            trial_parameters.key,
        )
    )

    sw = StopWatch()

    def log_to_file():
        fname_base = "{}_{}_{}_{}".format(
            fn.name,
            optimizer.qualified_name,
            trial_parameters.key,
            optimizer.evaluation_count,
        )
        if log_parameters.log_fitness:
            fname = f"{fname_base}.fitness.csv"
            np.savetxt(
                str(log_parameters.path.joinpath(fname).absolute()),
                optimizer.best.fitness,
                delimiter=",",
                header=header.format(
                    optimizer.evaluation_count,
                    sw.duration,
                    "fitness",
                ),
            )
        if log_parameters.log_points:
            fname = f"{fname_base}.points.csv"
            np.savetxt(
                str(log_parameters.path.joinpath(fname).absolute()),
                optimizer.best.points,
                delimiter=",",
                header=header.format(
                    optimizer.evaluation_count,
                    sw.duration,
                    "point",
                ),
            )
        if log_parameters.log_step_sizes:
            fname = f"{fname_base}.step_sizes.csv"
            np.savetxt(
                str(log_parameters.path.joinpath(fname).absolute()),
                optimizer.best.step_size,
                delimiter=",",
                header=header.format(
                    optimizer.evaluation_count,
                    sw.duration,
                    "step_size",
                ),
            )

    # Log initial points
    log_to_file()

    sw.start()
    while not optimizer.stop.triggered:
        points = optimizer.ask()
        if fn.has_constraints:
            fitness = fn.evaluate_with_penalty(points)
            optimizer.tell(*fitness)
        else:
            fitness = fn(points)
            optimizer.tell(fitness)
        if optimizer.evaluation_count in log_parameters.log_at:
            sw.stop()
            log_to_file()
            sw.start()

    return "{}-{}-{}".format(
        fn.name, optimizer.qualified_name, trial_parameters.key
    )


def log_mocma_trials(
    log_parameters: LogParameters,
    trial_parameters: List[MOCMATrialParameters],
    trial_slice: Optional[slice] = None,
    *,
    seed: int = 0,
    n_trials: int = 5,
    n_processes: int = 5,
    chunksize: Optional[int] = None,
    verbose: bool = True,
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
    chunksize: optional
        Chunksize to use.
    verbose: optional
        Print informative texts.

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

    jobs = list(
        dataclasses.replace(data, seed=seed, key=trial_id + 1)
        for (data, trial_id), seed in seeded_trial_parameters[trial_slice]
    )

    n_jobs = len(jobs)
    n_processes = min(n_processes, n_jobs, os.cpu_count())
    if chunksize is None:
        chunksize = 1
        if n_processes > 1:
            chunksize = (n_jobs // n_processes) + (
                1 if n_jobs % n_processes != 0 else 0
            )
    if chunksize > n_jobs:
        chunksize = 1
    if verbose:
        print(f"Number of jobs: {n_jobs}\n")
        print(f"Number of processes: {n_processes}\n")
        print(f"Chunksize: {chunksize}\n")
        print(f"First job: {jobs[0]}\n")
        print(f"Last job: {jobs[-1]}\n")

    if not log_parameters.path.exists():
        if verbose:
            print("Creating output directory.")
        os.makedirs(log_parameters.path.resolve(), exist_ok=True)
    if n_processes > 1:
        if verbose:
            print(f"Running {n_jobs} job(s) using parallel execution.")
        with multiprocessing.Pool(processes=n_processes) as pool:
            results = pool.map(
                partial(log_mocma_trial, log_parameters),
                jobs,
                chunksize=chunksize,
            )
    else:
        if verbose:
            print(f"Running {n_jobs} job(s) using sequential execution")
        results = list(map(partial(log_mocma_trial, log_parameters), jobs))

    if verbose:
        print(f"Completed {len(results)} jobs.")
        for result in results:
            print(f"\t{result}")


LOG_FILENAME_REGEX = re.compile(
    r"^([\w]+)[_]([\w\(\)\-\+]+)[_]([\d]+)[_]([\d]+)\.([\w]+)\.csv$"
)


class TrialLog:
    """An object to load a CSV log file.

    Parameters
    ----------
    path
        Path of the log file.

    Properties
    ----------
    path
        Path of the log file.
    fn
        The objective function name.
    optimizer
        The optimizer name.
    trial
        The trial identifier.
    n_evaluations
        The number of evaluations.
    observation_name
        The observation name.

    """

    path: pathlib.Path
    fn: str
    optimizer: str
    trial: int
    n_evaluations: int
    observation: str

    def __init__(self, path: Union[str, pathlib.Path], *, lazy: bool = True):
        if not isinstance(path, pathlib.Path):
            self.path = pathlib.Path(path)
        else:
            self.path = path
        # Parse metadata from the filename.
        metadata = LOG_FILENAME_REGEX.match(self.path.name)
        if not metadata:
            raise ValueError("Invalid filename")
        # Set properties from metadata tuple.
        metadata = metadata.groups()
        self.fn = metadata[0]
        self.optimizer = metadata[1]
        self.trial = int(metadata[2])
        self.n_evaluations = int(metadata[3])
        self.observation = metadata[4]
        # Load log data into memory or defer loading until required.
        self._data = None
        if not lazy:
            self._data = np.genfromtxt(self.path, delimiter=",")

    @property
    def data(self):
        """The observed data."""
        if self._data is None:
            self._data = np.genfromtxt(self.path, delimiter=",")
        return self._data

    def __repr__(self):
        return (
            "TrialLog(fn={},optimizer={},trial={},"
            "n_evaluations={},observation={})"
        ).format(
            self.fn,
            self.optimizer,
            self.trial,
            self.n_evaluations,
            self.observation,
        )


def get_options_pattern(options: Optional[List[Union[str, int]]]) -> str:
    """Convert a list of options into a glob pattern."""

    if options is None:
        return ".+"

    pattern = "{}".format(
        r"|".join(
            [
                str(option)
                .replace("(", r"\(")
                .replace(")", r"\)")
                .replace("-", r"\-")
                .replace("+", r"\+")
                for option in options
            ]
        )
    )
    if len(options) > 1:
        return fr"({pattern})"
    return pattern


InputPath = Union[str, pathlib.Path]


def load_logs(
    paths: Union[List[InputPath], InputPath],
    *,
    fns: Optional[List[str]] = None,
    opts: Optional[List[str]] = None,
    trials: Optional[List[int]] = None,
    n_evaluations: Optional[List[int]] = None,
    observations: Optional[List[str]] = None,
    lazy: bool = True,
    search_subdirs: bool = True,
) -> List[TrialLog]:
    """Load CSV log files.

    Parameters
    ----------
    paths
        Base paths in which to search log files.
    fns: optional
        A list of function names. By default uses `*` greedy pattern.
    opts: optional
        A list of optimizer names. By default uses `*` greedy pattern.
    trials: optional
        A list of trial numbers. By default uses `*` greedy pattern.
    n_evaluations: optional
        A list of evaluations numbers. By default uses `*` greedy pattern.
    observations: optional
        A list of observation names. By default uses `*` greedy pattern.
    lazy: optional
        Lazily load observed data into memory.
    search_subdirs: optional
        Whether to search within subdirectories.

    Returns
    -------
    List[TrialLog]
        A list of log objects.
    """
    if not isinstance(paths, list):
        paths = [paths]

    processed_paths = []
    for path in paths:
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        processed_paths.append(path)

    fns = get_options_pattern(fns)
    opts = get_options_pattern(opts)
    trials = get_options_pattern(trials)
    n_evaluations = get_options_pattern(n_evaluations)
    observations = get_options_pattern(observations)
    filename_pattern = re.compile(
        rf"^{fns}[_]{opts}[_]{trials}[_]{n_evaluations}\.{observations}\.csv$"
    )
    prefix = "**/" if search_subdirs else ""
    logs = []
    for path in processed_paths:
        for file_path in path.glob(f"{prefix}*.csv"):
            if filename_pattern.match(file_path.name):
                logs.append(TrialLog(file_path, lazy=lazy))
    return logs


__all__ = [
    "StopWatch",
    "MOCMATrialParameters",
    "UPMOCMATrialParameters",
    "LogParameters",
    "log_mocma_trials",
    "load_logs",
]
