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
        """
        return x * self.mask

    def transpose(self, y):
        """
        For the inpainting operator (a diagonal operator), the adjoint is the same as the forward operation.
        """
        return y * self.mask

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
        """
        Implements the proximal operator for ADMM.
        For a diagonal inpainting operator, a natural update is:
        
            x = (mask * y + rho * x) / (mask + rho)
        """
        return (self.mask * y + rho * x) / (self.mask + rho)

    def initialize(self, gt, y):
        """
        A simple initialization: use the measurements y for observed pixels and fill the missing parts with 0.
        """
        x_init = y.clone()
        x_init[self.mask == 0] = 0.0
        return x_init
