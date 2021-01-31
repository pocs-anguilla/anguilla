"""This module contains abstract classes modeling optimizers."""
import abc
import dataclasses

import numpy as np

from typing import Any, Callable, Iterable, Optional, Union, Tuple

try:
    from typing import final
except ImportError:
    from typing_extensions import final

OptimizableFunctionResult = Union[
    float,
    np.ndarray,
    Tuple[np.ndarray, np.ndarray, int],
    Tuple[np.ndarray, dict],
]
OptimizableFunction = Callable[[np.ndarray, Any], OptimizableFunctionResult]


class OptimizerSolution(metaclass=abc.ABCMeta):
    """Solution of an optimizer."""


class OptimizerParameters(metaclass=abc.ABCMeta):
    """Parameters for an optimizer."""


class OptimizerStoppingConditions(metaclass=abc.ABCMeta):
    """Stopping conditions for an optimizer."""


@dataclasses.dataclass
class OptimizerResult:
    """Result of an optimization run."""

    solution: OptimizerSolution
    stopping_conditions: OptimizerStoppingConditions


class Optimizer(metaclass=abc.ABCMeta):
    """Define interface of an optimizer.

    Notes
    -----
    It is based on the ask-and-tell pattern presented in \
    :cite:`2013:oo-optimizers` and implemented by :cite:`2019:pycma`.
    """

    # Set to True if tell should receive the search points.
    # This way the default implementation of fmin knows how to call tell.
    _tell_points: bool = False

    @property
    @abc.abstractmethod
    @final
    def name(self) -> str:
        """Return the name of the optimizer."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    @final
    def qualified_name(self) -> str:
        """Return the qualified name of the optimizer."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def stop(self) -> OptimizerStoppingConditions:
        """Return if any conditions trigger a stop."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def best(self) -> OptimizerSolution:
        """Return the current best solution."""
        raise NotImplementedError()

    @abc.abstractmethod
    def ask(self, *args: Any, **kwargs: Any) -> Any:
        """Produce offspring."""
        raise NotImplementedError()

    @abc.abstractmethod
    def tell(self, *args: Any, **kwargs: Any) -> None:
        """Pass offspring information to the optimizer."""
        raise NotImplementedError()

    def fmin(
        self,
        fn: OptimizableFunction,
        fn_args: Optional[Iterable[Any]] = None,
        fn_kwargs: Optional[dict] = None,
    ) -> OptimizerResult:
        """Optimize a function.

        Parameters
        ----------
        fn
            The function to optimize.
        fn_args: optional
            Positional arguments to pass when calling the function.
        fn_kwargs: optional
            Keyword arguments to pass when calling the function.

        Returns
        -------
        OptimizerResult
            The result of the optimization.

        Raises
        ------
        ValueError
            The function returns something that is not a \
            `OptimizableFunctionResult`.
        """
        if fn_args is None:
            fn_args = ()
        if fn_kwargs is None:
            fn_kwargs = {}

        while not self.stop.triggered:
            points = self.ask()
            result = fn(points, *fn_args, **fn_kwargs)
            # Determine which kind of result we're given
            if isinstance(result, (float, np.ndarray)):
                if not self._tell_points:
                    self.tell(result)
                else:
                    self.tell(points, result)
            elif isinstance(result, tuple):
                # Narrow down by duck typing
                if len(result) > 1 and isinstance(result[1], dict):
                    if not self._tell_points:
                        self.tell(result[0], **result[1])
                    else:
                        self.tell(points, result[0], **result[1])
                else:
                    if not self._tell_points:
                        self.tell(*result)
                    else:
                        self.tell(points, *result)
            else:
                raise ValueError(
                    "Return value must have type OptimizableFunctionResult"
                )
        return OptimizerResult(self.best, self.stop)


__all__ = [
    "OptimizableFunction",
    "OptimizableFunctionResult",
    "OptimizerSolution",
    "OptimizerParameters",
    "OptimizerStoppingConditions",
    "OptimizerResult",
    "Optimizer",
]
