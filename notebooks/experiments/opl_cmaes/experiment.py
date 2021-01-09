import dataclasses
import math
from typing import Callable, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from anguilla.optimizers.oplcma import OnePlusLambdaCMA, OPLStoppingConditions


@dataclasses.dataclass
class ExperimentConfiguration:
    """Hold experiment configuration."""

    n_dimensions: int
    ns_offspring: List[int]
    initial_region: Tuple[float]
    initial_step_size: float = 1e-4
    n_trials: int = 51
    max_evaluations: int = 1000
    target_fitness: Optional[float] = None
    seed: Optional[int] = None


class ExperimentResults:
    """Hold experiment results."""

    def __init__(
        self,
        fn_name: str,
        n_dimensions: int,
        ns_offspring: List[int],
        max_generations: int = 1,
    ) -> None:
        self.fn_name = fn_name
        self.n_dimensions = n_dimensions
        self.ns_offspring = ns_offspring
        self.history = np.zeros((len(ns_offspring), max_generations + 1)) - 1.0
        self.fitness = np.zeros((len(ns_offspring), 3)) - 1.0
        self.generations = np.zeros((len(ns_offspring), 3), dtype=int) - 1
        self.evaluations = np.zeros((len(ns_offspring), 3), dtype=int) - 1


def run_experiment(get_fn: Callable, config: ExperimentConfiguration):
    """Run an experiment."""
    max_generations = []
    for n_offspring in config.ns_offspring:
        max_generations.append(math.ceil(config.max_evaluations / n_offspring))
    fn = get_fn(2, None)
    results = ExperimentResults(
        fn.name,
        config.n_dimensions,
        config.ns_offspring,
        np.max(max_generations),
    )
    for i, n_offspring in enumerate(config.ns_offspring):
        n_trials = config.n_trials

        stopping_conditions = OPLStoppingConditions(
            max_evaluations=config.max_evaluations,
            target_fitness=config.target_fitness,
        )
        history = np.zeros((n_trials, max_generations[i])) - 1.0
        generations = np.zeros(n_trials, dtype=int) - 1
        fitness = np.zeros(n_trials) - 1.0
        seed = config.seed
        for trial in range(n_trials):
            if seed is not None:
                rng = np.random.default_rng(seed)
                seed += 1
            else:
                rng = np.random.default_rng()
            fn = get_fn(config.n_dimensions, rng)
            parent_point = fn.random_points(
                1, region_bounds=config.initial_region
            )
            parent_fitness = fn(parent_point)
            optimizer = OnePlusLambdaCMA(
                parent_point,
                parent_fitness,
                n_offspring=n_offspring,
                initial_step_size=config.initial_step_size,
                stopping_conditions=stopping_conditions,
                rng=rng,
            )
            while not optimizer.stop.triggered:
                history[
                    trial, optimizer.generation_count
                ] = optimizer.best.fitness
                points = optimizer.ask()
                if fn.has_constraints:
                    optimizer.tell(points, *fn.evaluate_with_penalty(points))
                else:
                    optimizer.tell(points, fn(points))
            history[
                trial, optimizer.generation_count - 1
            ] = optimizer.best.fitness
            generations[trial] = optimizer.generation_count
            fitness[trial] = optimizer.best.fitness

        sorted_idx = np.argsort(generations)
        results_idx = [
            sorted_idx[i]
            for i in (
                math.ceil(0.05 * n_trials),
                n_trials // 2,
                math.ceil(0.95 * n_trials),
            )
        ]
        results.history[i, 0 : generations[results_idx[1]]] = history[
            results_idx[1], 0 : generations[results_idx[1]]
        ]
        results.generations[i] = generations[results_idx]
        results.evaluations[i] = results.generations[i] * n_offspring
        results.fitness[i] = fitness[results_idx]
    return results


def plot_experiment(
    results: ExperimentResults,
    yscale: str = "log",
    xscale: str = "linear",
    xticks: Optional[np.ndarray] = None,
    yticks: Optional[np.ndarray] = None,
):
    """Plot an experiment."""
    fig = plt.figure(figsize=(4, 3.5))
    ax = fig.subplots(1, 1)
    if xticks is not None:
        ax.set_xlim(xticks[0], xticks[-1])
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_ylim(yticks[0], yticks[-1])
        ax.set_yticks(yticks)
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 4))
    ax.grid()

    for i, n_offspring in enumerate(results.ns_offspring):
        evaluations = np.arange(0, results.generations[i, 1]) * n_offspring
        fitness = results.history[i, 0 : results.generations[i, 1]]
        p = ax.plot(
            evaluations,
            fitness,
            label=f"(1+{n_offspring})-ES",
        )
        color = p[0].get_color()
        ax.plot(
            results.evaluations[i, [0, 2]],
            results.fitness[i, [0, 2]],
            color=color,
            marker="o",
            mfc="none",
            markersize=6,
            linestyle="dotted",
            label=None,
        )
        ax.plot(
            results.evaluations[i, 1],
            results.fitness[i, 1],
            color=color,
            marker="o",
            mfc="none",
            markersize=10,
            label=None,
        )

    ax.legend()
    fig.suptitle(f"{results.fn_name}, n={results.n_dimensions}")
    fig.tight_layout()
    return fig
