"""
Neural network architectures for Physics-Informed Neural Networks (PINNs).
"""

import torch
import torch.nn as nn


class FNN(nn.Module):
    """
    Fully Connected Neural Network with customizable architecture.
    
    Args:
        input_dim (int): Dimension of input features
        output_dim (int): Dimension of output features
        hidden_layers (list): List of integers specifying the number of neurons in each hidden layer
        activation (torch.nn.Module): Activation function to use between layers (default: Tanh)
    """
    
    def __init__(self, input_dim, output_dim, hidden_layers, activation=nn.Tanh()):
        super(FNN, self).__init__()
        
        # Input layer
        layers = [nn.Linear(input_dim, hidden_layers[0]), activation]
        
        # Hidden layers
        for i in range(1, len(hidden_layers)):
            layers.extend([
                nn.Linear(hidden_layers[i-1], hidden_layers[i]),
                activation
            ])
        
        # Output layer (no activation)
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        
        # Create the model
        self.model = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)
