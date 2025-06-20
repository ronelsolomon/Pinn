"""
Problem definitions for Physics-Informed Neural Networks (PINNs).

This package contains implementations of various PDE problems that can be solved using PINNs.
"""

from .base import Problem  # noqa: F401
from .burgers import BurgersProblem  # noqa: F401
from .heat import HeatEquation  # noqa: F401

__all__ = ['Problem', 'BurgersProblem', 'HeatEquation']
