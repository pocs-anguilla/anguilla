"""Testsuite for the :py:mod:`dominance` module."""
import dataclasses
import unittest
import typing

import numpy as np

from anguilla.dominance import non_dominated_sort, dominates, union_upper_bound


def line(m, b):
    """Compute a line"""
    return lambda x: m * x + b


def curve(a, b):
    """Compute a quadratic curve"""
    return lambda x: a * x * x + b


def get_objective_points(domain, *fns):
    """Evaluate the functions on the domain to create a population."""
    size = len(domain) * len(fns)
    points = np.empty((size, 2))
    i = 0
    for fn in fns:
        points[i : i + len(domain), 0] = domain
        points[i : i + len(domain), 1] = fn(domain)
        i += len(domain)
    return points


class TestNonDominatedSort(unittest.TestCase):
    """Test non-dominated sort."""

    def setUp(self) -> None:
        """Initialize state."""
        self._rng = np.random.default_rng(0)

    def test_all_ranks_different(self) -> None:
        """Test that all ranks are different."""
        # Create a line with positive unitary slope
        # Each point to the right in the x dimension
        # is dominated by all points to the left
        size = 10
        points = get_objective_points(np.arange(0, size), line(1, 0))
        ranks, max_rank = non_dominated_sort(points)
        self.assertTrue(
            np.all(ranks == np.arange(1, size + 1, dtype=int)), "Ranks"
        )
        self.assertTrue(max_rank == 10, "Max. rank")

    def test_all_ranks_equal(self) -> None:
        """Test that all ranks are equal."""
        # Create a line with negative unitary slope
        # Values increase monotonically in the x dimension
        # and decrease monotonically in the y dimension
        # So no point dominates another
        size = 10
        points = get_objective_points(np.arange(0, size), line(-1, 0))
        ranks, max_rank = non_dominated_sort(points)
        self.assertTrue(np.all(ranks == np.repeat(1, size)), "Ranks")
        self.assertTrue(max_rank == 1, "Max. rank")

    def test_three_translated_lines(self) -> None:
        """Test ranks of three lines with same slope but different bias"""
        # Create three lines with the same slope but different bias
        # Sort the points wrt the x dimension
        # The rank of the first point is 1 and the last is size * 2 + 1
        size = 5
        true_max_rank = size * 2 + 1
        points = get_objective_points(
            np.arange(0, size), line(1, 2), line(1, 3), line(1, 4)
        )
        sorted_points = np.array(sorted(points, key=lambda x: (x[0], x[1])))
        ranks, max_rank = non_dominated_sort(sorted_points)
        self.assertTrue(ranks[0] == 1, "First rank")
        self.assertTrue(ranks[-1] == true_max_rank, "Last rank")
        self.assertTrue(max_rank == true_max_rank, "Max. rank")
        for i in range(0, size * 3, 3):
            self.assertTrue(ranks[i] + 1 == ranks[i + 1], "Next point")
            self.assertTrue(ranks[i] + 2 == ranks[i + 2], "Second next point")
            if i > 0:
                self.assertTrue(ranks[i] == ranks[i - 1], "Previous point")

    def test_curve_and_line(self) -> None:
        """Test the ranks of points created with a curve and a line."""
        # Sort the points wrt x dimension
        # The first and last point have the minimum and maximum ranks
        # Every next two points have the same rank
        start = 5
        end = 20
        true_max_rank = end - start + 1
        points = get_objective_points(
            np.arange(start, end), curve(1, 0), line(1, 0)
        )
        sorted_points = np.array(sorted(points, key=lambda x: (x[0], x[1])))
        ranks, max_rank = non_dominated_sort(sorted_points)
        self.assertTrue(ranks[0] == 1, "First rank")
        self.assertTrue(ranks[-1] == true_max_rank, "Last rank")
        self.assertTrue(max_rank == true_max_rank, "Max. rank")
        for i in range(1, end - start, 2):
            self.assertTrue(ranks[i - 1] + 1 == ranks[i], "Prev. rank")
            self.assertTrue(ranks[i] == ranks[i + 1], "Peer rank")
            self.assertTrue(ranks[i] + 1 == ranks[i + 2], "Next rank")

    def test_shuffled(self) -> None:
        """Test the ranks of the shuffled points are the same."""
        points = self._rng.normal(size=(151, 2))
        ranks, max_rank = non_dominated_sort(points)
        shuffled_idx = np.arange(151, dtype=int)
        self._rng.shuffle(shuffled_idx)
        ranks_shuffled, max_rank_shuffled = non_dominated_sort(
            points[shuffled_idx]
        )
        self.assertTrue(max_rank == max_rank_shuffled, "Max. rank")
        self.assertTrue(np.all(ranks[shuffled_idx] == ranks_shuffled), "Ranks")


class TestDominates(unittest.TestCase):
    """Test dominates."""

    def test_dominated(self):
        self.assertTrue(
            dominates(np.array([1.0, 0.0, 1.5]), np.array([1.0, 0.0, 2.0]))
        )

    def test_notdominated(self):
        self.assertFalse(
            dominates(np.array([1.0, 0.1, 1.5]), np.array([1.0, 0.0, 2.0]))
        )

    def test_random(self):
        rng = np.random.default_rng()
        point1 = rng.uniform(size=32)
        self.assertFalse(dominates(point1, point1))


class TestUnionUpperBound(unittest.TestCase):
    def test_union_upper_bound(self):
        a = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        b = (np.array([5.0, 6.0]), np.array([3.0, 1.0]))
        c = (np.array([0.5, 4.0]), np.array([3.0, 3.0]))
        reference = union_upper_bound(a, b, c, translate_by=1.0)
        self.assertTrue(np.all(reference == np.array([4.0, 5.0])))
