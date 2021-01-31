"""This package contains implementations of single and multi-objective \
benchmark functions."""
from anguilla.fitness.benchmark.sphere import Sphere
from anguilla.fitness.benchmark.sum_squares import SumSquares
from anguilla.fitness.benchmark.ellipsoid import Ellipsoid
from anguilla.fitness.benchmark.rastrigin import Rastrigin
from anguilla.fitness.benchmark.elli import ELLI1, ELLI2, GELLI
from anguilla.fitness.benchmark.cigtab import CIGTAB1, CIGTAB2
from anguilla.fitness.benchmark.fon import FON
from anguilla.fitness.benchmark.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT4P, ZDT6
from anguilla.fitness.benchmark.dtlz import (
    DTLZ1,
    DTLZ2,
    DTLZ3,
    DTLZ4,
    DTLZ5,
    DTLZ6,
    DTLZ7,
)
from anguilla.fitness.benchmark.ihr import IHR1, IHR2, IHR3, IHR4, IHR6
from anguilla.fitness.benchmark.moq import MOQ

__all__ = [
    "Sphere",
    "SumSquares",
    "Ellipsoid",
    "Rastrigin",
    "ELLI1",
    "ELLI2",
    "GELLI",
    "CIGTAB1",
    "CIGTAB2",
    "FON",
    "ZDT1",
    "ZDT2",
    "ZDT3",
    "ZDT4",
    "ZDT4P",
    "ZDT6",
    "IHR1",
    "IHR2",
    "IHR3",
    "IHR4",
    "IHR6",
    "DTLZ1",
    "DTLZ2",
    "DTLZ3",
    "DTLZ4",
    "DTLZ5",
    "DTLZ6",
    "DTLZ7",
    "MOQ",
]
