"""This module contains implementations of special function evaluators."""
from anguilla.fitness.base import AbstractFunctionEvaluator


class PenalizedFunctionEvaluator(AbstractFunctionEvaluator):
    """Evaluate constrained functions with a penalty."""


class NoisyFunctionEvaluator(AbstractFunctionEvaluator):
    """Evaluate noisy functions."""
