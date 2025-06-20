"""
Implementation of the 1D Heat Equation for PINNs.

u_t - k * u_xx = 0, x in [0, 1], t in [0, 1]
u(x, 0) = sin(pi * x)  # Initial condition
u(0, t) = u(1, t) = 0  # Boundary conditions
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .base import Problem


class HeatEquation(Problem):
    """1D Heat Equation problem definition."""
    
    def __init__(self, k=0.1):
        """
        Initialize the Heat Equation problem.
        
        Args:
            k (float): Thermal diffusivity coefficient
        """
        super().__init__()
        self.k = k
        self.dim = 1  # 1D spatial dimension + time
        self.domains = {
            'x': (0.0, 1.0),  # Spatial domain
            't': (0.0, 1.0)   # Time domain
        }
    
    def sample_domain(self, n_points, val_split=0.1):
        """Sample points from the domain and split into train/validation sets."""
        # Sample points in the interior
        n_interior = int(n_points * 0.7)
        x_interior = np.random.uniform(self.domains['x'][0], self.domains['x'][1], (n_interior, 1))
        t_interior = np.random.uniform(self.domains['t'][0], self.domains['t'][1], (n_interior, 1))
        X_interior = np.hstack((x_interior, t_interior))
        
        # Sample points on the boundary
        n_boundary = n_points - n_interior
        t_boundary = np.random.uniform(self.domains['t'][0], self.domains['t'][1], (n_boundary, 1))
        X_boundary1 = np.hstack((np.ones_like(t_boundary) * self.domains['x'][0], t_boundary))  # x = 0
        X_boundary2 = np.hstack((np.ones_like(t_boundary) * self.domains['x'][1], t_boundary))  # x = 1
        X_boundary = np.vstack((X_boundary1, X_boundary2))
        
        # Sample points at initial condition
        x_initial = np.random.uniform(self.domains['x'][0], self.domains['x'][1], (n_boundary, 1))
        t_initial = np.zeros_like(x_initial)
        X_initial = np.hstack((x_initial, t_initial))
        
        # Combine all points
        X = np.vstack((X_interior, X_boundary, X_initial))
        
        # Split into train/validation
        n_val = int(len(X) * val_split)
        idx = np.random.permutation(len(X))
        train_idx, val_idx = idx[n_val:], idx[:n_val]
        
        return X[train_idx], X[val_idx]
    
    def pde_residual(self, x, u, model):
        """Compute the PDE residual for the Heat Equation."""
        # Enable gradient computation
        x.requires_grad_(True)
        
        # Get the prediction and its gradients
        u = model(x)
        
        # Compute first derivatives
        du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               create_graph=True, retain_graph=True)[0]
        du_dx = du[:, 0:1]  # du/dx
        du_dt = du[:, 1:2]  # du/dt
        
        # Compute second derivative
        d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx),
                                     create_graph=True, retain_graph=True)[0][:, 0:1]
        
        # Heat equation: u_t - k * u_xx = 0
        residual = du_dt - self.k * d2u_dx2
        
        return residual
    
    def boundary_condition_loss(self, model):
        """Compute the boundary condition loss (Dirichlet BCs)."""
        device = next(model.parameters()).device
        
        # Sample points on the boundary
        n_points = 100
        t = torch.linspace(self.domains['t'][0], self.domains['t'][1], n_points, device=device)
        
        # Left boundary (x = 0)
        x_left = torch.zeros_like(t)
        X_left = torch.stack([x_left, t], dim=1)
        u_pred_left = model(X_left)
        
        # Right boundary (x = 1)
        x_right = torch.ones_like(t)
        X_right = torch.stack([x_right, t], dim=1)
        u_pred_right = model(X_right)
        
        # Boundary condition: u(0, t) = u(1, t) = 0
        loss = torch.mean(u_pred_left**2) + torch.mean(u_pred_right**2)
        
        return loss
    
    def initial_condition_loss(self, model):
        """Compute the initial condition loss."""
        device = next(model.parameters()).device
        
        # Sample points at t = 0
        n_points = 100
        x = torch.linspace(self.domains['x'][0], self.domains['x'][1], n_points, device=device)
        t = torch.zeros_like(x)
        X = torch.stack([x, t], dim=1)
        
        # Get model prediction and exact solution
        u_pred = model(X)
        u_initial = torch.sin(np.pi * x).view(-1, 1)
        
        return torch.mean((u_pred - u_initial)**2)
    
    def exact_solution(self, x):
        """
        Return the exact solution to the Heat Equation.
        
        For the problem u_t = k * u_xx with u(x,0) = sin(pi*x) and u(0,t) = u(1,t) = 0,
        the exact solution is u(x,t) = sin(pi*x) * exp(-k*pi^2*t)
        """
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.array(x)
            
        if len(x_np.shape) == 1:
            x_vals = x_np[0]
            t_vals = x_np[1]
        else:
            x_vals = x_np[:, 0]
            t_vals = x_np[:, 1]
            
        return np.sin(np.pi * x_vals) * np.exp(-self.k * np.pi**2 * t_vals)
    
    def plot_solution(self, model, save_path=None):
        """Plot the solution as a surface plot."""
        device = next(model.parameters()).device
        
        # Create a grid of points
        x = np.linspace(self.domains['x'][0], self.domains['x'][1], 100)
        t = np.linspace(self.domains['t'][0], self.domains['t'][1], 100)
        X, T = np.meshgrid(x, t)
        
        # Flatten and predict
        x_flat = X.flatten()
        t_flat = T.flatten()
        X_test = np.vstack((x_flat, t_flat)).T
        
        # Get exact solution if available
        exact_sol = self.exact_solution(X_test)
        
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
            u_pred = model(X_test_tensor).cpu().numpy()
        
        # Reshape for plotting
        U_pred = u_pred.reshape(X.shape)
        U_exact = exact_sol.reshape(X.shape)
        
        # Create the plot
        fig = plt.figure(figsize=(15, 5))
        
        # Plot predicted solution
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(X, T, U_pred, cmap='viridis', linewidth=0, antialiased=False)
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
        ax1.set_zlabel('u(x,t)')
        ax1.set_title('PINN Solution')
        
        # Plot exact solution if available
        if exact_sol is not None:
            ax2 = fig.add_subplot(122, projection='3d')
            surf2 = ax2.plot_surface(X, T, U_exact, cmap='viridis', linewidth=0, antialiased=False)
            ax2.set_xlabel('x')
            ax2.set_ylabel('t')
            ax2.set_zlabel('u(x,t)')
            ax2.set_title('Exact Solution')
            
            # Add a color bar
            fig.colorbar(surf2, ax=[ax1, ax2], shrink=0.5, aspect=10)
        else:
            fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig, (ax1, ax2) if 'ax2' in locals() else (ax1,)
