import torch

class InpaintingMask:
    def __init__(self, height, width, crop_height=None, crop_width=None,
                 corner_top=None, corner_left=None, channels=3, device='cuda'):
        """
        Initializes the inpainting mask generator.

        Args:
            height (int): Image height.
            width (int): Image width.
            crop_height (int, optional): Height of the missing region. Defaults to half of height.
            crop_width (int, optional): Width of the missing region. Defaults to half of width.
            corner_top (int, optional): Top coordinate of the missing region. Defaults to height//4.
            corner_left (int, optional): Left coordinate of the missing region. Defaults to 45% of width.
            channels (int): Number of channels. Typically 1 or 3.
            device (str): Torch device.
        """
        self.height = height
        self.width = width
        self.channels = channels
        self.device = device

        self.crop_height = crop_height if crop_height is not None else height // 2
        self.crop_width = crop_width if crop_width is not None else width // 2
        self.corner_top = corner_top if corner_top is not None else height // 4
        self.corner_left = corner_left if corner_left is not None else int(0.45 * width)
        
        self.mask = self._create_mask()
        
    def _create_mask(self):
        # Create a mask with ones (observed pixels)
        mask = torch.ones((1, self.channels, self.height, self.width), device=self.device)
        # Set the missing region to zeros (unobserved pixels)
        mask[:, :, self.corner_top:self.corner_top+self.crop_height, 
                  self.corner_left:self.corner_left+self.crop_width] = 0.0
        return mask

    def get_mask(self):
        return self.mask
