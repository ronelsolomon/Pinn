"""
Core package for Physics-Informed Neural Networks (PINNs).

This package contains the main PINN implementation, neural network architectures,
and training utilities.
"""

from .network import FNN  # noqa: F401
from .losses import PINNLoss  # noqa: F401
from .trainer import PINN  # noqa: F401

__all__ = ['FNN', 'PINNLoss', 'PINN']
