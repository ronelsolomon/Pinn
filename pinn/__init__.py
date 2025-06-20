"""
Physics-Informed Neural Networks (PINNs) for solving differential equations.

This package provides tools for solving partial differential equations (PDEs)
using deep learning with physics-informed constraints.
"""

from pinn.core import PINN
from pinn.problems import BurgersProblem, HeatEquation

__version__ = '0.1.0'
__all__ = ['PINN', 'BurgersProblem', 'HeatEquation']
