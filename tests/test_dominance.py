"""Testsuite for the :py:mod:`dominance` module."""
import dataclasses
import unittest
import typing

import numpy as np

from anguilla.dominance import (
    fast_non_dominated_sort,
    naive_non_dominated_sort,
)


@dataclasses.dataclass
class Unit:
    """A container of a unit test example."""

    ps: np.ndarray
    ranks: np.ndarray
    target_size: typing.Optional[int] = None


class TestDominanceOperators(unittest.TestCase):
    """Unit tests for dominance operators."""

    def setUp(self) -> None:
        """Initialize state."""
        rng = np.random.default_rng(0)
        units = []
        ps = rng.uniform(size=(3, 2))
        units.append(Unit(ps=ps, ranks=np.array([2, 1, 3])))
        ps = rng.uniform(size=(3, 2))
        units.append(Unit(ps=ps, ranks=np.array([1, 1, 1])))
        ps = rng.uniform(size=(5, 2))
        units.append(Unit(ps=ps, ranks=np.array([1, 2, 3, 2, 1])))
        units.append(
            Unit(ps=ps, ranks=np.array([1, 0, 0, 0, 1]), target_size=2)
        )
        units.append(
            Unit(ps=ps, ranks=np.array([1, 2, 0, 2, 1]), target_size=3)
        )
        ps = rng.uniform(size=(1, 1))
        units.append(Unit(ps=ps, ranks=np.array([1])))
        self.units = units

    def test_against_units(self) -> None:
        """Test against precomputed ranks."""
        for unit in self.units:
            ranks, _ = fast_non_dominated_sort(
                unit.ps, target_size=unit.target_size
            )
            self.assertTrue(np.all(ranks == unit.ranks))

    def test_against_naive(self) -> None:
        """Test against naive implementation."""
        for unit in self.units:
            if unit.target_size is not None:
                ranks, _ = fast_non_dominated_sort(unit.ps)
                ranks_naive = naive_non_dominated_sort(unit.ps)
                self.assertTrue(np.all(ranks == ranks_naive))
