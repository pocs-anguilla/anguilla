"""This package contains implementations of single and multi-objective \
benchmark functions."""
from anguilla.fitness.benchmark.sphere import Sphere
from anguilla.fitness.benchmark.sum_squares import SumSquares
from anguilla.fitness.benchmark.elli import ELLI1
from anguilla.fitness.benchmark.fon import FON
from anguilla.fitness.benchmark.zdt import ZDT4P
from anguilla.fitness.benchmark.dtlz import DTLZ1, DTLZ2
from anguilla.fitness.benchmark.ihr import IHR1, IHR2, IHR3, IHR4, IHR6

__all__ = [
    "Sphere",
    "SumSquares",
    "ELLI1",
    "FON",
    "ZDT4P",
    "DTLZ1",
    "DTLZ2",
    "IHR1",
    "IHR2",
    "IHR3",
    "IHR4",
    "IHR6",
]
