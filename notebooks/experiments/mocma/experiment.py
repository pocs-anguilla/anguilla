import os
import dataclasses
import time
import multiprocessing
from typing import Iterable, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

from anguilla.fitness.base import ObjectiveFunction
from anguilla.optimizers.mocma import MOCMA


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
        """Elapsed wall-clock time in seconds."""
        return self._duration


@dataclasses.dataclass
class TrialDuration:
    """Keep score of total wall-clock time per operation type in a trial.

    Params
    ------
    ask
        Total wall-clock time elapsed for ask operations.
    tell
        Total wall-clock time elapsed for tell operations.
    eval
        Total wall-clock time elapsed for function evaluations.
    """

    ask: float
    tell: float
    eval: float

    @property
    def total(self):
        """Total wall-clock time elapsed for all measured operations."""
        return self.ask + self.tell + self.eval


@dataclasses.dataclass
class TrialParameters:
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
class TrialResult:
    initial_fitness: np.ndarray
    final_fitness: np.ndarray
    volume: float
    reference: np.ndarray
    generation_count: int
    evaluation_count: int
    duration: TrialDuration
    fn_name: str
    optimizer_name: str
    parameters: TrialParameters


def run_trial(parameters: TrialParameters):
    if parameters.seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(parameters.seed)
    if parameters.fn_rng_seed is not None:
        parameters.fn_kwargs["rng"] = np.random.default_rng(
            parameters.fn_rng_seed
        )
    else:
        parameters.fn_kwargs["rng"] = rng
    fn = parameters.fn_cls(*parameters.fn_args, **parameters.fn_kwargs)
    if not fn.has_scalable_objectives and fn.n_objectives != 2:
        raise ValueError("The provided function does not support 2 objectives")
    fn.n_objectives = 2
    parent_points = fn.random_points(
        parameters.n_parents, region_bounds=parameters.region_bounds
    )
    parent_fitness = fn(parent_points)
    optimizer = MOCMA(
        parent_points,
        parent_fitness,
        n_offspring=parameters.n_offspring,
        rng=rng,
        success_notion=parameters.success_notion,
        max_generations=parameters.max_generations,
        max_evaluations=parameters.max_evaluations,
        target_indicator_value=parameters.target_indicator_value,
    )
    if parameters.reference is not None:
        optimizer.indicator.reference = parameters.reference
    initial_fitness = optimizer.best.fitness
    ask_sw = StopWatch()
    tell_sw = StopWatch()
    eval_sw = StopWatch()
    while not optimizer.stop.triggered:
        ask_sw.start()
        points = optimizer.ask()
        ask_sw.stop()
        if fn.has_constraints:
            eval_sw.start()
            fitness = fn.evaluate_with_penalty(points)
            eval_sw.stop()
            tell_sw.start()
            optimizer.tell(*fitness)
            tell_sw.stop()
        else:
            eval_sw.start()
            fitness = fn(points)
            eval_sw.stop()
            tell_sw.start()
            optimizer.tell(fitness)
            tell_sw.stop()
    final_fitness = optimizer.best.fitness
    volume = None
    if parameters.reference is not None:
        volume = optimizer.indicator(final_fitness)
    duration = TrialDuration(
        ask_sw.duration, tell_sw.duration, eval_sw.duration
    )
    return TrialResult(
        initial_fitness=initial_fitness,
        final_fitness=final_fitness,
        volume=volume,
        reference=parameters.reference,
        generation_count=optimizer.generation_count,
        evaluation_count=optimizer.evaluation_count,
        duration=duration,
        fn_name=fn.qualified_name,
        optimizer_name=optimizer.qualified_name,
        parameters=parameters,
    )


def run_trials(
    parameters: TrialParameters,
    seed: int = 0,
    n_trials: int = 5,
    n_processes: int = 5,
) -> Iterable[TrialResult]:
    """Run independent trials of MOCMA with a benchmark function.

    Parameters
    ----------
    parameters
        Parameters for each trial.
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
    seeds = seed_sequence.spawn(n_trials)

    params = (
        dataclasses.replace(parameters, seed=seeds[trial], key=trial + 1)
        for trial in range(n_trials)
    )
    n_processes = min(n_processes, n_trials)
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
            results = pool.map(run_trial, params, chunksize=chunksize)
    else:
        print(
            "Running {} trial(s) using sequential execution".format(n_trials)
        )
        results = list(map(run_trial, params))

    return results


def runtime_summary(results: Iterable[TrialResult]) -> None:
    """Summarize operation duration in a set of trials."""
    times = np.zeros(4)
    samples = 0
    for result in results:
        samples += 1
        times[0] += result.duration.total
        times[1] += result.duration.ask
        times[2] += result.duration.tell
        times[3] += result.duration.eval
    times /= samples
    print(
        """Average wall-clock time: total = {:.2f}s, ask = {:.2f}s, """
        """tell = {:.2f}s, eval = {:.2f}s""".format(*times)
    )


def volume_summary(results: Iterable[TrialResult]) -> None:
    """Summarize obtained volumes."""
    if results[0].parameters.reference is not None:
        n_trials = len(results)
        volumes = np.zeros(n_trials)
        for trial, result in enumerate(results):
            volumes[trial] = result.volume
        target_value = results[0].parameters.target_indicator_value
        if target_value is not None:
            target_value = "{:.6E}".format(target_value)
            print(
                """HV with reference point ({}) and target value ({}):"""
                """\n max. = {:.6E}, median = {:.6E}, min. = {:.6E}""".format(
                    results[0].parameters.reference,
                    target_value,
                    np.max(volumes),
                    np.median(volumes),
                    np.min(volumes),
                )
            )
    else:
        print("No reference point was provided.")


def pareto_front_plot_2d(
    fn: ObjectiveFunction,
    num: int = 50,
    xscale="linear",
    yscale="linear",
) -> Any:
    """Plot the Pareto front of a bi-objective benchmark function.

    Parameters
    ----------
    fn
        The benchmark function.
    num: optional
        The number of samples.
    xscale: optional
        The scale of the first axis.
    yscale:
        The scale of the second axis.

    Returns
    -------
    Any
        The figure object.
    """
    fn.n_objectives = 2
    fig = plt.figure(figsize=(4, 4))
    ax = fig.subplots(1, 1)
    front = fn.pareto_front(num)
    ax.plot(front[0], front[1])
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    fig.tight_layout()
    return fig


def population_plot_2d(
    results: Iterable[TrialResult],
    plot_initial_fronts: bool = True,
    plot_true_front: bool = True,
    sharex: bool = False,
    sharey: bool = False,
    xscale: str = "linear",
    yscale: str = "linear",
) -> Any:
    """Make a 2D population plot of independent runs of a MO optimizer.

    Parameters
    ----------
    results
        Results of the run trials.
    plot_initial_fronts
        Plot the initial front of the benchmark function.
    plot_true_front
        Plot the true Pareto front if the benchmark function implements it.
    sharex
        Share the first axis (relevant if `plot_initial_fronts` is `True`).
    sharey
        Share the second axis.
    """
    if results[0].parameters.fn_rng_seed is not None:
        results[0].parameters.fn_kwargs["rng"] = np.random.default_rng(
            results[0].parameters.fn_rng_seed
        )
    fn = results[0].parameters.fn_cls(
        *results[0].parameters.fn_args, **results[0].parameters.fn_kwargs
    )
    fn.n_objectives = 2
    if plot_initial_fronts:
        fig = plt.figure(figsize=(8, 4))
        ax0, ax1 = fig.subplots(1, 2, sharey=sharey, sharex=sharex)
    else:
        fig = plt.figure(figsize=(4, 4))
        ax0, ax1 = fig.subplots(1, 1), None
    if plot_true_front and fn.has_known_pareto_front:
        front = fn.pareto_front()
        if fn.has_continuous_pareto_front:
            if plot_initial_fronts:
                ax1.plot(
                    front[0],
                    front[1],
                    color="red",
                    linestyle="-",
                    marker=None,
                    label="True",
                )
            ax0.plot(
                front[0],
                front[1],
                color="red",
                linestyle="-",
                marker=None,
                label="True",
            )
        else:
            if plot_initial_fronts:
                ax1.plot(
                    front[0],
                    front[1],
                    color="red",
                    linestyle="",
                    marker=".",
                    label="True",
                )
            ax0.plot(
                front[0],
                front[1],
                color="red",
                linestyle="",
                marker=".",
                label="True",
            )

    color = None
    n_trials = len(results)
    volumes = np.zeros(n_trials)
    generations = np.zeros(n_trials)
    evaluations = np.zeros(n_trials)
    for trial, result in enumerate(results):
        generations[trial] = result.generation_count
        evaluations[trial] = result.evaluation_count
        if result.parameters.reference is not None:
            volumes[trial] = result.volume
        if plot_initial_fronts:
            p = ax1.plot(
                result.initial_fitness[:, 0],
                result.initial_fitness[:, 1],
                marker="x",
                linestyle="",
            )
            color = p[0].get_color()
        if color is not None:
            ax0.plot(
                result.final_fitness[:, 0],
                result.final_fitness[:, 1],
                marker="x",
                linestyle="",
                color=color,
                label=f"Trial {result.parameters.key}",
            )
            color = None
        else:
            ax0.plot(
                result.final_fitness[:, 0],
                result.final_fitness[:, 1],
                marker="x",
                linestyle="",
                label=f"Trial {result.parameters.key}",
            )

    fig.suptitle(
        """Population plot for {}: $\mathbb{{R}}^{{{}}} \mapsto """
        """\mathbb{{R}}^{}$ using {}\n for {} avg. generations /"""
        """ {} avg. evaluations in {} trials""".format(
            fn.qualified_name,
            fn.n_dimensions,
            fn.n_objectives,
            results[0].optimizer_name,
            int(np.average(generations)),
            int(np.average(evaluations)),
            n_trials,
        )
    )
    if results[0].parameters.reference is not None:
        ax0.set_title(
            "Final front\n(max. HV = {:.6E})".format(np.max(volumes))
        )
    else:
        ax0.set_title("Final front\nNo reference")
    ax0.set_ylabel("Second objective")
    ax0.set_xlabel("First objective")
    ax0.set_xscale(xscale)
    ax0.set_yscale(yscale)
    if plot_initial_fronts:
        ax1.set_title("Initial front\n(random points)")
        ax1.set_xscale(xscale)
        ax1.set_yscale(yscale)

    ax0.legend()
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    from anguilla.fitness import benchmark

    parameters = TrialParameters(
        benchmark.ZDT1,
        fn_args=(30,),
        max_evaluations=50000,
        n_offspring=1,
    )

    sw = StopWatch()
    sw.start()
    results = run_trials(parameters, seed=12345, n_trials=1)
    sw.stop()
    print(f"Total wall-clock time: {sw.duration:.2f}")
    runtime_summary(results)
    fig = population_plot_2d(results)
    fig.savefig("test.png")
