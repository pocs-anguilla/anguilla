"""Testsuite for the :py:mod:`selection` package."""
import unittest
from typing import Any, Optional, Tuple

import numpy as np

from anguilla.selection import indicator_selection
from anguilla.indicators import HypervolumeIndicator


class TestHypervolumeIndicatorSelection(unittest.TestCase):
    """Test hypervolume-indicator-based selection"""

    def test_example1(self):
        """Test a constructed 2D example"""
        # A set of non-dominated points
        points = np.array(
            [
                [0.01245897, 0.27127751],
                [0.02213313, 0.23395707],
                [0.0233907, 0.22994154],
                [0.0392689, 0.1886141],
                [0.04339422, 0.17990426],
                [0.16521067, 0.05107939],
                [0.17855283, 0.0440614],
                [0.28619405, 0.00950565],
            ]
        )
        expected = np.array(
            [True, False, False, False, False, True, False, True], dtype=bool
        )
        indicator = HypervolumeIndicator()
        selected, _ = indicator_selection(indicator, points, len(points) - 5)
        self.assertTrue(np.all(selected == expected))

    def test_example2(self):
        """Test a constructed 2D example"""
        # A set of non-dominated points
        # A line with negative slope
        points = np.array(
            [
                [0.0, 0.0],
                [1.0, -1.0],
                [2.0, -2.0],
                [3.0, -3.0],
                [4.0, -4.0],
                [5.0, -5.0],
                [6.0, -6.0],
                [7.0, -7.0],
                [8.0, -8.0],
                [9.0, -9.0],
            ]
        )
        expected = np.array(
            [True, False, True, False, True, False, True, False, False, True],
            dtype=bool,
        )
        indicator = HypervolumeIndicator()
        selected, _ = indicator_selection(indicator, points, len(points) - 5)
        self.assertTrue(np.all(selected == expected))
