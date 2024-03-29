"""Testsuite for the :py:mod:`hypervolume` package."""
import math
import unittest
import pathlib

import numpy as np

from typing import Any, Optional

from anguilla.hypervolume import (
    calculate,
    contributions,
    contributions_naive,
    hvkd_f8,
)

from anguilla.util import (
    random_2d_3d_front,
    random_cliff_3d,
    random_convex_front,
    random_concave_front,
    random_linear_front,
)


class HVBaseTestFunction:

    in_filename: str
    out_filename: str
    assertTrue: Any

    def setUp(self):
        in_path = (
            pathlib.Path(__file__)
            .parent.joinpath("data/contributions")
            .joinpath(self.in_filename)
            .absolute()
        )
        out_path = (
            pathlib.Path(__file__)
            .parent.joinpath("data/contributions")
            .joinpath(self.out_filename)
            .absolute()
        )
        in_data = np.genfromtxt(str(in_path), delimiter=",")
        self.reference = in_data[0]
        if np.any(np.isnan(self.reference)):
            self.reference = None
        self.points = in_data[1:]
        out_data = np.genfromtxt(str(out_path), delimiter=",")
        self.volume = out_data[0]
        self.contributions = out_data[1:].T
        self.rng = np.random.default_rng()

    def test_volume(self):
        vol = calculate(self.points, self.reference)
        self.assertTrue(math.isclose(self.volume, vol, rel_tol=1e-6))

    def test_contributions(self):
        contribs = contributions(self.points, self.reference)
        self.assertTrue(np.allclose(self.contributions, contribs))


class TestSharkTC2D(HVBaseTestFunction, unittest.TestCase):
    """Unit test from Shark."""

    in_filename = "SHARK_2D_IN.csv"
    out_filename = "SHARK_2D_OUT.csv"


class TestSharkTC3D(HVBaseTestFunction, unittest.TestCase):
    """Unit test from Shark."""

    in_filename = "SHARK_3D_IN.csv"
    out_filename = "SHARK_3D_OUT.csv"


class TestCliff3D1(HVBaseTestFunction, unittest.TestCase):
    """Random points using Cliff3D."""

    in_filename = "CLIFF3D_1_IN.csv"
    out_filename = "CLIFF3D_1_OUT.csv"


class TestHVKD(unittest.TestCase):
    """Hypervolume calculation of k-D fronts."""

    def test_random_cliff_3d(self):
        front = random_cliff_3d(25)
        nadir = np.max(front, axis=0)
        vol_a = calculate(front, nadir)
        vol_b = hvkd_f8(front, nadir)
        self.assertTrue(
            math.isclose(vol_a, vol_b),
            "expected: {}, got: {}".format(vol_a, vol_b),
        )

    def test_random_convex_3d(self):
        front = random_convex_front(25, 3)
        nadir = np.max(front, axis=0)
        vol_a = calculate(front, nadir)
        vol_b = hvkd_f8(front, nadir)
        self.assertTrue(
            math.isclose(vol_a, vol_b),
            "expected: {}, got: {}".format(vol_a, vol_b),
        )

    def test_random_concave_3d(self):
        front = random_concave_front(25, 3)
        nadir = np.max(front, axis=0)
        vol_a = calculate(front, nadir)
        vol_b = hvkd_f8(front, nadir)
        self.assertTrue(
            math.isclose(vol_a, vol_b),
            "expected: {}, got: {}".format(vol_a, vol_b),
        )

    def test_random_linear_3d(self):
        front = random_linear_front(25, 3)
        nadir = np.max(front, axis=0)
        vol_a = calculate(front, nadir)
        vol_b = hvkd_f8(front, nadir)
        self.assertTrue(
            math.isclose(vol_a, vol_b),
            "expected: {}, got: {}".format(vol_a, vol_b),
        )


class TestOther(unittest.TestCase):
    """Other HV unit tests."""

    def test_2d_naive(self) -> None:
        """Test the 2-D implementation using the naive implementation."""
        _, _, front, nadir = random_2d_3d_front(10, dominated=False)
        contrib_a = contributions_naive(front, nadir)
        contrib_b = contributions(front, nadir)
        self.assertTrue(np.allclose(contrib_a, contrib_b))

    def test_2d_3d(self) -> None:
        """Test the 3-D implementation using the simplest of the 2-D."""
        front_3d, nadir_3d, front_2d, nadir_2d = random_2d_3d_front(
            10, dominated=False
        )
        contrib_a = contributions(front_3d, nadir_3d)
        contrib_b = contributions(front_2d, nadir_2d)
        self.assertTrue(np.allclose(contrib_a, contrib_b))

    def test_auto_refpoint_computation(self) -> None:
        points = random_cliff_3d(250)
        reference = np.max(points, axis=0)
        contrib_a = contributions(points, reference)
        contrib_b = contributions(points)
        self.assertTrue(np.allclose(contrib_a, contrib_b))
