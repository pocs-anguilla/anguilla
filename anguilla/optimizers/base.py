"""This module contains abstract classes modeling optimizers."""
import abc
import dataclasses


from typing import Any, Callable, Iterable, Optional

try:
    from typing import final
except ImportError:
    from typing_extensions import final


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

    @abc.abstractmethod
    def fmin(
        self,
        fn: Callable,
        fn_args: Optional[Iterable[Any]] = None,
        fn_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> OptimizerResult:
        """Optimize the given function."""
        raise NotImplementedError()
