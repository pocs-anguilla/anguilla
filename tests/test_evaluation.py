"""Testsuite for the :py:mod:`evaluation` module."""
import unittest
import time
import math

from anguilla.evaluation import StopWatch


class TestStopWatch(unittest.TestCase):
    """Test the stop watch."""

    def test_basic(self):
        """Test basic operation."""
        sw = StopWatch()
        sw.start()
        time.sleep(0.35)
        sw.stop()
        self.assertTrue(
            math.isclose(sw.duration, 0.35, abs_tol=2),
            "Got: {}, Expected: {}".format(sw.duration, 0.35),
        )
