import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import register_operator, LinearOperator

@register_operator(name='inpainting')
class Inpainting(LinearOperator):
    def __init__(self, mask, device):
        """
        Args:
            mask (torch.Tensor): A binary tensor of shape (B, C, H, W) with 1 for observed pixels
                              and 0 for missing ones.
            device (torch.device): The device to which the mask and operations are sent.
        """
        self.mask = mask.to(device)
        self.device = device
        
    @property
    def display_name(self):
        return 'inpainting'
    
    def forward(self, x, **kwargs):
        """
        Applies the inpainting operator H to x by element-wise multiplication with the mask.
        For inpainting, H takes only observed pixels, so H(x) = mask * x
        """
        return self.mask * x
    
    def transpose(self, y):
        """
        For the inpainting operator, the adjoint H^T is also multiplication by the mask.
        H^T takes measurements and places them at observed pixel locations.
        """
        return self.mask * y
    
    def proximal_generator(self, x, y, sigma, rho):
        """
        Implements a proximal generator that samples from the Gaussian conditional distribution
        using the fact that H is a binary (diagonal) operator.
        
        For each pixel i, the update is:
        
            x_i = (mask_i * y_i/sigma^2 + x_i/rho^2) / (mask_i/sigma^2 + 1/rho^2) + noise_i
        
        where noise_i ~ N(0, 1/(mask_i/sigma^2 + 1/rho^2)).
        """
        # Compute the inverse variance elementwise: 
        # 1 / (mask/sigma^2 + 1/rho^2)
        inv_var = 1 / (self.mask / (sigma**2) + 1 / (rho**2))
        noise = torch.sqrt(inv_var) * torch.randn_like(x)
        mu_x = inv_var * (self.mask * y / (sigma**2) + x / (rho**2))
        return mu_x + noise
    
    def proximal_for_admm(self, x, y, rho):
    
        raise NotImplementedError("This method is not used in the current implementation")
    
    def initialize(self, gt, y):
        """
        Initialize the solution with measurements at observed locations
        and zeros at unobserved locations.
        
        Args:
            gt (torch.Tensor): Ground truth image (not used, kept for API consistency)
            y (torch.Tensor): Observed measurements
            
        Returns:
            torch.Tensor: Initial guess for the reconstruction
        """
        x_init = y.clone()
        x_init[self.mask == 0] = 0.0
        return x_init