"""
Loss functions for Physics-Informed Neural Networks (PINNs).
"""

import torch
import torch.nn as nn


class PINNLoss(nn.Module):
    """
    Combined loss function for Physics-Informed Neural Networks.
    
    Combines data loss (if any) with physics loss (PDE residual).
    """
    
    def __init__(self):
        super(PINNLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, y_pred, y_true=None, f_pred=None, 
                data_weight=1.0, pde_weight=1.0, bc_weight=1.0, ic_weight=1.0):
        """
        Compute the combined loss.
        
        Args:
            y_pred: Model predictions
            y_true: Ground truth values (None if no data loss)
            f_pred: PDE residuals (None if no physics loss)
            data_weight: Weight for data loss term
            pde_weight: Weight for PDE residual loss term
            bc_weight: Weight for boundary condition loss term
            ic_weight: Weight for initial condition loss term
            
        Returns:
            Combined loss value
        """
        total_loss = 0.0
        
        # Data loss (if ground truth is provided)
        if y_true is not None:
            data_loss = self.mse_loss(y_pred, y_true)
            total_loss += data_weight * data_loss
        
        # Physics loss (PDE residual)
        if f_pred is not None:
            pde_loss = torch.mean(f_pred**2)
            total_loss += pde_weight * pde_loss
        
        # Note: Boundary and initial condition losses would be added here
        # They are typically implemented as part of the specific problem definition
        
        return total_loss
