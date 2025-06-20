"""
Example script for solving the 1D Burgers' equation using a Physics-Informed Neural Network (PINN).

u_t + u * u_x - nu * u_xx = 0, x in [-1, 1], t in [0, 1]
u(x, 0) = -sin(pi * x)
u(-1, t) = u(1, t) = 0
"""

import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

# Add the project root to the Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinn.core import PINN
from pinn.problems.burgers import BurgersProblem


def train_burgers():
    """Train a PINN to solve the Burgers' equation."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create the problem
    problem = BurgersProblem(nu=0.01/np.pi)
    
    # Initialize the PINN
    input_dim = 2  # x and t
    output_dim = 1  # u(x,t)
    hidden_layers = [20, 20, 20, 20, 20]
    
    pinn = PINN(input_dim, output_dim, hidden_layers, activation=nn.Tanh())
    
    # Set up the optimizer
    optimizer = optim.Adam
    lr = 0.001
    epochs = 10000
    
    print("Training PINN for Burgers' equation...")
    print(f"Network architecture: {hidden_layers}")
    print(f"Learning rate: {lr}")
    print(f"Number of epochs: {epochs}")
    
    # Train the model
    pinn.train(
        problem=problem,
        optimizer=lambda params: optimizer(params, lr=lr),
        epochs=epochs,
        n_points=2000,
        val_split=0.1,
        verbose=True
    )
    
    return pinn, problem


def plot_training_history(pinn, save_dir='results'):
    """Plot the training history."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.semilogy(pinn.history['train_loss'], label='Training Loss')
    plt.semilogy(pinn.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Train the model
    pinn, problem = train_burgers()
    
    # Plot training history
    plot_training_history(pinn)

    # Create and save animation
    print("Creating animation...")
    problem.animate_solution(pinn, save_path='results/burgers_animation.gif', fps=10)
    
    # Plot and save the solution
    problem.plot_solution(pinn, save_path='results/burgers_solution.png')
    
   
    
    # Save the model
    pinn.save('results/burgers_model.pt')
    print("Model saved to 'results/burgers_model.pt'")


if __name__ == "__main__":
    main()
