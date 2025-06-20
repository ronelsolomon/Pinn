"""
Training utilities for Physics-Informed Neural Networks (PINNs).
"""

import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from .network import FNN
from .losses import PINNLoss


class PINN:
    """
    Physics-Informed Neural Network (PINN) implementation.
    
    This class handles the training and evaluation of PINN models.
    """
    
    def __init__(self, input_dim, output_dim, hidden_layers, activation=torch.nn.Tanh()):
        """
        Initialize the PINN.
        
        Args:
            input_dim (int): Dimension of input features
            output_dim (int): Dimension of output features
            hidden_layers (list): List of integers specifying the number of neurons in each hidden layer
            activation: Activation function to use between layers
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FNN(input_dim, output_dim, hidden_layers, activation).to(self.device)
        self.loss_fn = PINNLoss()
        self.optimizer = None
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train(self, problem, optimizer, epochs=1000, batch_size=32, 
              n_points=1000, val_split=0.1, verbose=True):
        """
        Train the PINN on a given problem.
        
        Args:
            problem: Problem instance defining the PDE and domain
            optimizer: Optimizer to use for training
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            n_points (int): Number of collocation points to sample
            val_split (float): Fraction of data to use for validation
            verbose (bool): Whether to show progress bar
        """
        self.optimizer = optimizer(self.model.parameters())
        
        # Generate training and validation data
        X_train, X_val = problem.sample_domain(n_points, val_split)
        
        # Convert to PyTorch tensors and move to device
        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device, requires_grad=True)
        X_val = torch.tensor(X_val, dtype=torch.float32, device=self.device, requires_grad=True)
        
        # Training loop
        best_loss = float('inf')
        for epoch in tqdm(range(epochs), disable=not verbose):
            self.model.train()
            
            # Forward pass
            u_pred = self.model(X_train)
            
            # Compute physics loss (PDE residual)
            f_pred = problem.pde_residual(X_train, u_pred, self.model)
            
            # Compute boundary/initial condition losses
            bc_loss = problem.boundary_condition_loss(self.model)
            ic_loss = problem.initial_condition_loss(self.model)
            
            # Combine losses
            loss = self.loss_fn(
                y_pred=u_pred,
                f_pred=f_pred,
                pde_weight=1.0,
                bc_weight=1.0,
                ic_weight=1.0
            )
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Validation
            self.model.eval()
            
            # Get predictions without tracking gradients
            with torch.no_grad():
                u_val = self.model(X_val)
            
            # Prepare for gradient computation
            x_val = X_val.detach().requires_grad_(True)
            u_val = u_val.detach().requires_grad_(True)
            
            # Compute residual with gradients
            f_val = problem.pde_residual(x_val, u_val, self.model)
            
            # Compute validation loss without tracking gradients
            with torch.no_grad():
                val_loss = self.loss_fn(
                    y_pred=u_val.detach(),
                    f_pred=f_val.detach(),
                    pde_weight=1.0,
                    bc_weight=1.0,
                    ic_weight=1.0
                )
            
            # Save history
            self.history['train_loss'].append(loss.item())
            self.history['val_loss'].append(val_loss.item())
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = self.model.state_dict()
        
        # Load best model
        self.model.load_state_dict(best_model_state)
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input tensor or numpy array
            
        Returns:
            Model predictions as numpy array
        """
        self.model.eval()
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)
        with torch.no_grad():
            y_pred = self.model(X)
        return y_pred.cpu().numpy()
    
    def save(self, path):
        """Save the model to a file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
    
    def load(self, path):
        """Load the model from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})
