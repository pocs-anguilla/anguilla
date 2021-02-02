"""Testsuite for the :py:mod:`evaluation` module."""
import unittest

import numpy as np

from anguilla.evaluation import union_upper_bound


class TestUnionUpperBound(unittest.TestCase):
    def test_union_upper_bound(self):
        a = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        b = (np.array([5.0, 6.0]), np.array([3.0, 1.0]))
        c = (np.array([0.5, 4.0]), np.array([3.0, 3.0]))
        reference = union_upper_bound(a, b, c, translate_by=1.0)
        self.assertTrue(np.all(reference == np.array([4.0, 5.0])))
