"""Testsuite for the :py:mod:`fitness.benchmark` module."""
import unittest
from typing import Any

import numpy as np

from anguilla.fitness.base import AbstractObjectiveFunction
from anguilla.fitness.benchmark import DTLZ1, DTLZ2


class BaseTestFunction:
    """Base test case for benchmark functions.

    Notes
    -----
    The implementation follows the approach suggested \
    `here <https://stackoverflow.com/a/52261358>`_.
    """

    fn: AbstractObjectiveFunction
    xs: np.ndarray
    ys: np.ndarray

    assertTrue: Any

    def test_single(self):
        for x, y in zip(self.xs, self.ys):
            self.assertTrue(np.allclose(self.fn.evaluate_single(x), y))

    def test_multiple(self):
        self.assertTrue(
            np.allclose(self.fn.evaluate_multiple(self.xs), self.ys)
        )

    def test_single_multiple(self):
        for x in self.xs:
            v1 = self.fn.evaluate_single(x)
            v2 = self.fn.evaluate_multiple(x)
            self.assertTrue(v1.shape == v2.shape, "Shapes don't match.")
            self.assertTrue(np.allclose(v1, v2), "Values don't match.")

    def test_call(self):
        self.assertTrue(
            np.allclose(self.fn(self.xs), self.ys),
            "Call with multiple failed.",
        )
        for x, y in zip(self.xs, self.ys):
            self.assertTrue(
                np.allclose(self.fn(x), y), "Call with single failed."
            )


class TestDTLZ1(BaseTestFunction, unittest.TestCase):
    """Unit tests for the DTLZ1 function."""

    def setUp(self) -> None:
        self.fn = DTLZ1(5, 3)
        self.xs = np.array(
            [
                [0.135477, 0.83500859, 0.96886777, 0.22103404, 0.30816705],
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.2, 0.2, 0.5, 0.2, 0.2],
            ]
        )
        # These values were determined with Shark
        self.ys = np.array(
            [
                [14.72018423, 2.90859756, 112.49501215],
                [0.125, 0.125, 0.25],
                [0.38, 1.52, 7.6],
            ]
        )


class TestDTLZ2(BaseTestFunction, unittest.TestCase):
    """Unit tests for the DTLZ1 function."""

    def setUp(self) -> None:
        self.fn = DTLZ2(5, 2)
        self.xs = np.array(
            [
                [0.135477, 0.83500859, 0.96886777, 0.22103404, 0.30816705],
                [0.5472206, 0.18838198, 0.9928813, 0.99646133, 0.96769494],
                [0.72583896, 0.98110969, 0.10986175, 0.79810586, 0.29702945],
            ]
        )
        # These values were determined with Shark
        self.ys = np.array(
            [
                [1.41405515, 0.30554692],
                [1.17839815, 1.36759131],
                [0.6319292, 1.37552529],
            ]
        )
