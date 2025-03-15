import torch
import torch.nn.functional as F
from modules.inpainting_mask import InpaintingMask
from . import register_operator, LinearOperator

@register_operator(name='inpainting')
class Inpainting(LinearOperator):
    def __init__(self, height, width, device='cuda', **kwargs):
        # Generate the inpainting mask using our helper module.
        mask_gen = InpaintingMask(height, width, device=device, **kwargs)
        self.mask = mask_gen.get_mask()
        self.device = device

    @property
    def display_name(self):
        return 'inpainting'

    def forward(self, x, **kwargs):
        return x * self.mask

    def transpose(self, y):
        # For a binary mask, forward and transpose are identical.
        return y * self.mask

    def proximal_generator(self, x, y, sigma, rho):
        # For each pixel, compute Q = mask/sigma^2 + 1/rho^2 (diagonal operator)
        Q = self.mask / (sigma ** 2) + 1 / (rho ** 2)
        Q_inv = 1.0 / Q
        mu = Q_inv * (self.mask * y / (sigma ** 2) + x / (rho ** 2))
        noise = torch.randn_like(x) * torch.sqrt(Q_inv)
        return mu + noise

    def proximal_for_admm(self, x, y, rho):
        # A simple closed-form proximal operator for ADMM.
        return (self.mask * y + rho * x) / (self.mask + rho)

    def initialize(self, gt, y):
        return torch.zeros_like(gt)
