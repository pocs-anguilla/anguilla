"""Testsuite for the :py:mod:`evaluation` module."""
import unittest
import time
import math

from anguilla.fitness import benchmark
from anguilla.evaluation import (
    StopWatch,
    MOCMATrialParameters,
    UPMOCMATrialParameters,
    LogParameters,
)


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


class TestParameters(unittest.TestCase):
    """Test the parameter classes."""

    def test_mocma_initialization(self):
        """Test creating an instance."""
        MOCMATrialParameters(benchmark.ZDT1)

    def test_upmocma_initialization(self):
        """Test creating an instance."""
        UPMOCMATrialParameters(benchmark.ZDT1)

    def test_log_initialization(self):
        """Test creating an instance."""
        LogParameters("./logs", [10, 50])
