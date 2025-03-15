# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from . import register_operator, LinearOperator

# @register_operator(name='inpainting')
# class Inpainting(LinearOperator):
#     def __init__(self, mask, device):
#         """
#         Args:
#             mask (torch.Tensor): A binary tensor of shape (B, C, H, W) with 1 for observed pixels 
#                                  and 0 for missing ones.
#             device (torch.device): The device to which the mask and operations are sent.
#         """
#         self.mask = mask.to(device)
#         self.device = device

#     @property
#     def display_name(self):
#         return 'inpainting'

#     def forward(self, x, **kwargs):
#         """
#         Applies the inpainting operator H to x by element-wise multiplication with the mask.
#         """
#         return x * self.mask

#     def transpose(self, y):
#         """
#         For the inpainting operator (a diagonal operator), the adjoint is the same as the forward operation.
#         """
#         return y * self.mask

#     def proximal_generator(self, x, y, sigma, rho):
#         """
#         Implements a proximal generator that samples from the Gaussian conditional distribution
#         using the fact that H is a binary (diagonal) operator.
        
#         For each pixel i, the update is:
        
#             x_i = (mask_i * y_i/sigma^2 + x_i/rho^2) / (mask_i/sigma^2 + 1/rho^2) + noise_i
        
#         where noise_i ~ N(0, 1/(mask_i/sigma^2 + 1/rho^2)).
#         """
#         # Compute the inverse variance elementwise: 
#         # 1 / (mask/sigma^2 + 1/rho^2)
#         inv_var = 1 / (self.mask / (sigma**2) + 1 / (rho**2))
#         noise = torch.sqrt(inv_var) * torch.randn_like(x)
#         mu_x = inv_var * (self.mask * y / (sigma**2) + x / (rho**2))
#         return mu_x + noise

#     def proximal_for_admm(self, x, y, rho):
#         """
#         Implements the proximal operator for ADMM.
#         For a diagonal inpainting operator, a natural update is:
        
#             x = (mask * y + rho * x) / (mask + rho)
#         """
#         return (self.mask * y + rho * x) / (self.mask + rho)

#     def initialize(self, gt, y):
#         """
#         A simple initialization: use the measurements y for observed pixels and fill the missing parts with 0.
#         """
#         x_init = y.clone()
#         x_init[self.mask == 0] = 0.0
#         return x_init

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
        using the Sherman-Morrison-Woodbury formula.
        
        For inpainting, H^T H is diagonal (since H selects a subset of rows from the identity),
        so Q_x^(-1) = σ²I_N - σ⁴H^T(I_M + σ²HH^T)^(-1)H = σ²I_N - σ⁴H^TH(I_M + σ²)^(-1)
        
        Since H^TH is the mask (diagonal), this simplifies to:
        Q_x^(-1) = σ²I_N for unobserved pixels, and σ²I_N - σ⁴/(σ² + 1) for observed pixels
                 = σ²I_N * (1 - mask * σ²/(σ² + 1))
        """
        # Implementation of the E-PO algorithm for inpainting
        
        # For inpainting, H^T H is diagonal with values equal to the mask
        # Using the Sherman-Morrison-Woodbury formula:
        # Q_x^(-1) = σ²I - σ⁴H^T(I_M + σ²HH^T)^(-1)H
        # Since H^TH is diagonal (mask), this simplifies
        
        # Compute the covariance matrix diagonal elements
        covariance = sigma**2 * (1 - self.mask * (sigma**2 / (sigma**2 + 1)))
        
        # Mean vector: μ_x = Q_x^(-1) * (H^T y/σ² + x/ρ²)
        # For observed pixels: (mask * y / σ² + x / ρ²) * covariance
        # For unobserved pixels: x / ρ² * σ²
        
        mean = covariance * (self.mask * y / sigma**2 + x / rho**2)
        
        # Sample from the Gaussian distribution
        noise = torch.sqrt(covariance) * torch.randn_like(x)
        return mean + noise
    
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