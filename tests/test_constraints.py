"""Testsuite for the :py:mod:`fitness.constraints` module."""
import unittest

import numpy as np

from anguilla.fitness.constraints import BoxConstraintsHandler


class TestBoxConstraintsHandler(unittest.TestCase):
    """Test the box constraints handler."""

    def test_closest_feasible_simple(self):
        """Test with simple bounds tuple."""
        n = 3
        lower = -10.0
        upper = 10.0
        handler = BoxConstraintsHandler(n, (lower, upper))
        # All coordinates outside the lower bound
        input = np.array([-11.0, -21.0, -30.0])
        computed = handler.closest_feasible(input)
        expected = np.repeat(lower, n)
        self.assertTrue(
            np.all(computed == expected),
            "Lower bound, got: {}, expected: {}".format(computed, expected),
        )
        # All coordinates outside the upper bound
        input = np.array([11.0, 21.0, 31.0])
        computed = handler.closest_feasible(input)
        expected = np.repeat(upper, n)
        self.assertTrue(
            np.all(computed == expected),
            "Upper bound, got: {}, expected: {}".format(computed, expected),
        )
        # All coordinates within the bounds
        input = np.array([0.0, 1.0, -3.0])
        computed = handler.closest_feasible(input)
        expected = input
        self.assertTrue(
            np.all(computed == expected),
            "Within bounds, got: {}, expected: {}".format(computed, expected),
        )
        # Mixed: some within and some out of bounds
        input = np.array([-12.0, -5.0, 5.0, 56.0])
        computed = handler.closest_feasible(input)
        expected = np.array([lower, -5.0, 5.0, upper])
        self.assertTrue(
            np.all(computed == expected),
            "Mixed bag, got: {}, expected: {}".format(computed, expected),
        )

    def test_closest_feasible_custom(self):
        """Test with custom bounds tuple."""
        lower = np.array([-10.0, -20.0, -30.0, -5.0])
        upper = np.array([10.0, 20.0, 30.0, 40.0])
        handler = BoxConstraintsHandler(4, (lower, upper))
        # All coordinates outside the lower bound
        input = np.array([-11.0, -21.0, -31.0, -5.0])
        computed = handler.closest_feasible(input)
        expected = lower
        self.assertTrue(
            np.all(computed == expected),
            "Lower bound, got: {}, expected: {}".format(computed, expected),
        )
        # All coordinates outside the upper bound
        input = np.array([11.0, 21.0, 31.0, 454.0])
        computed = handler.closest_feasible(input)
        expected = upper
        self.assertTrue(
            np.all(computed == expected),
            "Upper bound, got: {}, expected: {}".format(computed, expected),
        )
        # All coordinates within the bounds
        input = np.array([0.0, 1.0, -3.0, 0.0])
        computed = handler.closest_feasible(input)
        expected = input
        self.assertTrue(
            np.all(computed == expected),
            "Within bounds, got: {}, expected: {}".format(computed, expected),
        )
        # Mixed: some within and some out of bounds
        input = np.array([-12.0, -5.0, 5.0, 56.0])
        computed = handler.closest_feasible(input)
        expected = np.array([-10.0, -5.0, 5.0, 40.0])
        self.assertTrue(
            np.all(computed == expected),
            "Mixed bag, got: {}, expected: {}".format(computed, expected),
        )
