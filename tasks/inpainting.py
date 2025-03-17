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
#                               and 0 for missing ones.
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
#         using the Sherman-Morrison-Woodbury formula.
        
#         For inpainting, H^T H is diagonal (since H selects a subset of rows from the identity),
#         so Q_x^(-1) = σ²I_N - σ⁴H^T(I_M + σ²HH^T)^(-1)H = σ²I_N - σ⁴H^TH(I_M + σ²)^(-1)
        
#         Since H^TH is the mask (diagonal), this simplifies to:
#         Q_x^(-1) = σ²I_N for unobserved pixels, and σ²I_N - σ⁴/(σ² + 1) for observed pixels
#                  = σ²I_N * (1 - mask * σ²/(σ² + 1))
#         """
#         # Implementation of the E-PO algorithm for inpainting
        
#         # For inpainting, H^T H is diagonal with values equal to the mask
#         # Using the Sherman-Morrison-Woodbury formula:
#         # Q_x^(-1) = σ²I - σ⁴H^T(I_M + σ²HH^T)^(-1)H
#         # Since H^TH is diagonal (mask), this simplifies
        
#         # Compute the covariance matrix diagonal elements
#         covariance = sigma**2 * (1 - self.mask * (sigma**2 / (sigma**2 + 1)))
        
#         # Mean vector: μ_x = Q_x^(-1) * (H^T y/σ² + x/ρ²)
#         # For observed pixels: (mask * y / σ² + x / ρ²) * covariance
#         # For unobserved pixels: x / ρ² * σ²
        
#         mean = covariance * (self.mask * y / sigma**2 + x / rho**2)
        
#         # Sample from the Gaussian distribution
#         noise = torch.sqrt(covariance) * torch.randn_like(x)
#         return mean + noise
    
#     def proximal_for_admm(self, x, y, rho):
#         """
#         Implements the proximal operator for ADMM.
#         For a diagonal inpainting operator, a natural update is:
                
#         x = (mask * y + rho * x) / (mask + rho)
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
        For inpainting, H takes only observed pixels, so H(x) = mask * x
        """
        return self.mask * x
    
    def transpose(self, y):
        """
        For the inpainting operator, the adjoint H^T is also multiplication by the mask.
        H^T takes measurements and places them at observed pixel locations.
        """
        return self.mask * y
    
    # def proximal_generator(self, x, y, sigma, rho):
    #     """
    #     Implements exact perturbation-optimization (E-PO) algorithm for sampling from 
    #     the Gaussian conditional distribution p(x|z,y;ρ²) = N(x; μ_x, Q_x^(-1))
        
    #     Based on equation (17) and (18) from the paper:
    #     Q_x = (1/σ²)H^T H + (1/ρ²)I_N
    #     μ_x = Q_x^(-1) * ((1/σ²)H^T y + (1/ρ²)z)
        
    #     Using Sherman-Morrison-Woodbury formula in equation (18):
    #     Q_x^(-1) = ρ² * (I_N - (ρ²/(σ² + ρ²))H^T H)
        
    #     For inpainting, H^T H is diagonal with entries equal to the mask.
    #     """
    #     # Calculate Q_x^(-1) using equation (18)
    #     # For observed pixels (mask=1): ρ² * (1 - ρ²/(σ² + ρ²))
    #     # For missing pixels (mask=0): ρ²
    #     inv_Q_x = rho**2 * (1 - (rho**2 / (sigma**2 + rho**2)) * self.mask)
        
    #     # Calculate μ_x using equation (17)
    #     # μ_x = Q_x^(-1) * ((1/σ²)H^T y + (1/ρ²)z)
    #     # Where H^T y = mask * y and z = x (our input)
    #     mu_x = inv_Q_x * ((1/sigma**2) * self.mask * y + (1/rho**2) * x)
        
    #     # Sample from N(μ_x, Q_x^(-1))
    #     # Draw random noise and scale by sqrt of variance
    #     noise = torch.randn_like(x) * torch.sqrt(inv_Q_x)
        
    #     # Return sample from the conditional distribution
    #     return mu_x + noise
    
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
        This is a deterministic update for use in optimization methods.
        
        For inpainting, the proximal operator simplifies to a weighted average:
        x = (mask * y + rho * x) / (mask + rho)
        
        Where unobserved pixels (mask=0) remain unchanged and observed pixels
        become a weighted average of the measurement and current estimate.
        """
        # # Alternative form based on the same principles as E-PO, but without the noise
        # denominator = self.mask / sigma**2 + torch.ones_like(self.mask) / rho**2
        # x_update = (self.mask * y / sigma**2 + x / rho**2) / denominator
        # return x_update
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