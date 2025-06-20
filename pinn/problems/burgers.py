"""
Implementation of the 1D Burgers' equation for PINNs.

u_t + u * u_x - nu * u_xx = 0, x in [-1, 1], t in [0, 1]
u(x, 0) = -sin(pi * x)  # Initial condition
u(-1, t) = u(1, t) = 0  # Boundary conditions
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .base import Problem
import os
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


class BurgersProblem(Problem):
    """1D Burgers' equation problem definition."""
    
    def __init__(self, nu=0.01/np.pi):
        """
        Initialize the Burgers' equation problem.
        
        Args:
            nu (float): Viscosity coefficient
        """
        super().__init__()
        self.nu = nu
        self.dim = 1  # 1D spatial dimension + time
        self.domains = {
            'x': (-1.0, 1.0),  # Spatial domain
            't': (0.0, 1.0)    # Time domain
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
        x_boundary = np.random.choice(self.domains['x'], size=(n_boundary // 2, 1))
        t_boundary = np.random.uniform(self.domains['t'][0], self.domains['t'][1], (n_boundary // 2, 1))
        X_boundary1 = np.hstack((np.ones_like(t_boundary) * self.domains['x'][0], t_boundary))  # x = -1
        X_boundary2 = np.hstack((np.ones_like(t_boundary) * self.domains['x'][1], t_boundary))  # x = 1
        X_boundary = np.vstack((X_boundary1, X_boundary2))
        
        # Sample points at initial condition
        x_initial = np.random.uniform(self.domains['x'][0], self.domains['x'][1], (n_boundary // 2, 1))
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
        """Compute the PDE residual for Burgers' equation."""
        # Ensure x requires gradients and is detached from the current graph
        x = x.detach().requires_grad_(True)
        
        # Get the prediction
        u = model(x)
        
        # Compute first derivatives
        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )[0]
        
        du_dx = grad_u[..., 0:1]  # Spatial derivative
        du_dt = grad_u[..., 1:2]  # Temporal derivative
        
        # Compute second derivative
        grad_du_dx = torch.autograd.grad(
            outputs=du_dx,
            inputs=x,
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )[0]
        d2u_dx2 = grad_du_dx[..., 0:1]
        
        # Burgers' equation: u_t + u * u_x - nu * u_xx = 0
        residual = du_dt + u * du_dx - self.nu * d2u_dx2
        
        # Clean up to avoid memory leaks
        x.requires_grad_(False)
        
        return residual
    
    def boundary_condition_loss(self, model):
        """Compute the boundary condition loss (Dirichlet BCs)."""
        device = next(model.parameters()).device
        
        # Sample points on the boundary
        n_points = 100
        t = torch.linspace(self.domains['t'][0], self.domains['t'][1], n_points, device=device)
        
        # Left boundary (x = -1)
        x_left = torch.ones_like(t) * self.domains['x'][0]
        X_left = torch.stack([x_left, t], dim=1)
        u_pred_left = model(X_left)
        
        # Right boundary (x = 1)
        x_right = torch.ones_like(t) * self.domains['x'][1]
        X_right = torch.stack([x_right, t], dim=1)
        u_pred_right = model(X_right)
        
        # Boundary condition: u(-1, t) = u(1, t) = 0
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
        u_exact = self.exact_solution(X.detach().cpu().numpy())
        
        if u_exact is not None:
            u_exact = torch.tensor(u_exact, dtype=torch.float32, device=device)
            return torch.mean((u_pred - u_exact)**2)
        
        # If no exact solution, use the expected initial condition
        u_initial = -torch.sin(np.pi * x).view(-1, 1)
        return torch.mean((u_pred - u_initial)**2)
    
    def exact_solution(self, x):
        """
        Return the exact solution to Burgers' equation (if available).
        
        Note: For most nu values, the exact solution is not analytically available.
        This is a placeholder that returns None. For nu=0.01/pi, a reference solution
        can be computed numerically but is not implemented here.
        """
        return None
    
    def plot_solution(self, model, save_path=None):
        """Plot the solution as a surface plot."""
        # Handle both PINN and raw model objects
        if hasattr(model, 'model'):
            # If model is a PINN instance, get the underlying model
            device = next(model.model.parameters()).device
            predict = model.model
        else:
            # If model is a raw model
            device = next(model.parameters()).device
            predict = model
        
        # Create a grid of points
        x = np.linspace(self.domains['x'][0], self.domains['x'][1], 100)
        t = np.linspace(self.domains['t'][0], self.domains['t'][1], 100)
        X, T = np.meshgrid(x, t)
        
        # Flatten and predict
        x_flat = X.flatten()
        t_flat = T.flatten()
        X_test = np.vstack((x_flat, t_flat)).T
        
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
            u_pred = predict(X_test_tensor).cpu().numpy()
        
        # Reshape for plotting
        U = u_pred.reshape(X.shape)
        
        # Create the plot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, T, U, cmap='viridis', linewidth=0, antialiased=False)
        
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x,t)')
        ax.set_title('Burgers\' Equation Solution')
        
        # Add a color bar
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig, ax

    def animate_solution(self, model, save_path='results/burgers_animation.gif', fps=10):
        """
        Create an animation of the solution over time.
        
        Args:
            model: The trained PINN or model
            save_path: Path to save the animation (default: 'results/burgers_animation.gif')
            fps: Frames per second for the animation
        """
        print("Starting animation creation...")
        try:
            import matplotlib.animation as animation
            from matplotlib.animation import PillowWriter
        except ImportError as e:
            print(f"Error importing animation modules: {e}")
            return None
        
        # Handle both PINN and raw model objects
        try:
            if hasattr(model, 'model'):
                device = next(model.model.parameters()).device
                predict = model.model
            else:
                device = next(model.parameters()).device
                predict = model
            print(f"Using device: {device}")
        except Exception as e:
            print(f"Error setting up model: {e}")
            return None
        
        # Create spatial grid
        x = np.linspace(self.domains['x'][0], self.domains['x'][1], 100)
        print(f"Spatial grid: {len(x)} points from {x[0]} to {x[-1]}")
        
        # Create time points for animation
        t_values = np.linspace(self.domains['t'][0], self.domains['t'][1], 10)  # Reduced frames for testing
        print(f"Time points: {len(t_values)} from {t_values[0]} to {t_values[-1]}")
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Initialize line
        line, = ax.plot([], [], 'b-', linewidth=2, label='Predicted')
        
        # Add exact solution if available
        exact_line = None
        if hasattr(self, 'exact_solution'):
            exact_line, = ax.plot([], [], 'r--', linewidth=2, label='Exact')
        
        # Set up the axis
        ax.set_xlim([self.domains['x'][0], self.domains['x'][1]])
        ax.set_ylim([-1.5, 1.5])
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.set_title('Burgers\' Equation Solution Over Time')
        ax.legend()
        ax.grid(True)
        
        # Store data for debugging
        debug_data = []
        
        # Initialization function
        def init():
            line.set_data([], [])
            if exact_line is not None:
                exact_line.set_data([], [])
            return (line, exact_line) if exact_line is not None else (line,)
        
        # Animation function
        def animate(t):
            try:
                print(f"\rRendering frame at t = {t:.2f}", end="")
                # Create input tensor for this time step
                x_tensor = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, device=device)
                t_tensor = torch.full_like(x_tensor, t, device=device)
                X = torch.cat([x_tensor, t_tensor], dim=1)
                
                # Get prediction
                with torch.no_grad():
                    u_pred = predict(X).cpu().numpy()
                
                # Store for debugging
                debug_data.append((t, x, u_pred.copy()))
                
                # Update line data
                line.set_data(x, u_pred.flatten())
                
                # Update exact solution if available
                if exact_line is not None:
                    u_exact = self.exact_solution(np.column_stack((x, np.full_like(x, t))))
                    if u_exact is not None:
                        exact_line.set_data(x, u_exact)
                
                ax.set_title(f"Burgers' Equation Solution (t = {t:.2f})")
                
                return (line, exact_line) if exact_line is not None else (line,)
                
            except Exception as e:
                print(f"\nError in animation frame at t={t}: {e}")
                import traceback
                traceback.print_exc()
                return init()
        
        print("\nCreating animation...")
        try:
            # Create animation with blit=False for debugging
            anim = animation.FuncAnimation(
                fig, animate, frames=t_values,
                init_func=init, blit=False, interval=1000/fps
            )
            
            # Save the animation
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            print(f"\nSaving animation to {save_path}...")
            
            # Try different writers if Pillow fails
            try:
                writer = PillowWriter(fps=fps)
                anim.save(save_path, writer=writer, dpi=100)
            except Exception as e:
                print(f"Error with Pillow writer: {e}")
                print("Trying ffmpeg writer...")
                try:
                    writer = animation.FFMpegWriter(fps=fps)
                    anim.save(save_path.replace('.gif', '.mp4'), writer=writer)
                    save_path = save_path.replace('.gif', '.mp4')
                except Exception as e2:
                    print(f"Error with ffmpeg writer: {e2}")
                    print("Saving as HTML...")
                    from IPython.display import HTML
                    html = anim.to_html5_video()
                    with open(save_path.replace('.gif', '.html'), 'w') as f:
                        f.write(html)
                    save_path = save_path.replace('.gif', '.html')
            
            plt.close()
            print(f"Animation saved to {save_path}")
            
            # Save debug data
            if debug_data:
                import pickle
                debug_path = save_path + '.debug.pkl'
                with open(debug_path, 'wb') as f:
                    pickle.dump(debug_data, f)
                print(f"Debug data saved to {debug_path}")
                
            return anim
            
        except Exception as e:
            print(f"Error creating animation: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        finally:
            plt.close('all')
