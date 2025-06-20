"""
Base class for defining PDE problems to be solved with PINNs.
"""

import numpy as np
import torch


class Problem:
    """
    Base class for defining PDE problems for PINNs.
    
    This class provides a template for implementing specific PDE problems.
    Subclasses should implement the abstract methods to define the problem.
    """
    
    def __init__(self):
        """Initialize the problem with default parameters."""
        self.dim = None  # Spatial dimension of the problem
        self.domains = None  # Dictionary of domain bounds for each dimension
    
    def sample_domain(self, n_points, val_split=0.1):
        """
        Sample points from the problem domain.
        
        Args:
            n_points (int): Total number of points to sample
            val_split (float): Fraction of points to use for validation
            
        Returns:
            tuple: (train_points, val_points) as numpy arrays
        """
        raise NotImplementedError("Subclasses must implement sample_domain")
    
    def pde_residual(self, x, u, model):
        """
        Compute the PDE residual.
        
        Args:
            x: Input tensor with shape (n_points, dim+1) where last column is time
            u: Model predictions at x
            model: The neural network model
            
        Returns:
            PDE residual as a tensor
        """
        raise NotImplementedError("Subclasses must implement pde_residual")
    
    def boundary_condition_loss(self, model):
        """
        Compute the boundary condition loss.
        
        Args:
            model: The neural network model
            
        Returns:
            Boundary condition loss as a scalar tensor
        """
        return torch.tensor(0.0)  # Default: no boundary condition
    
    def initial_condition_loss(self, model):
        """
        Compute the initial condition loss.
        
        Args:
            model: The neural network model
            
        Returns:
            Initial condition loss as a scalar tensor
        """
        return torch.tensor(0.0)  # Default: no initial condition
    
    def exact_solution(self, x):
        """
        Return the exact solution at points x (if available).
        
        Args:
            x: Points at which to evaluate the exact solution
            
        Returns:
            Exact solution values at points x
        """
        return None  # Default: no exact solution available
    
    def plot_solution(self, model, save_path=None):
        """
        Plot the solution.
        
        Args:
            model: The trained model
            save_path: If provided, save the plot to this path
        """
        raise NotImplementedError("Subclasses should implement plot_solution")
