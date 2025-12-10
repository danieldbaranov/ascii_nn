import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from transformers import AutoModelForImageSegmentation
from torch import nn
import numpy as np


def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    """Preprocess a single image or batch of images into normalized model input."""
    if im.ndim == 3:
        im = im[np.newaxis, ...]  # (1,H,W,C)
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(0, 3, 1, 2)
    im_tensor = F.interpolate(im_tensor, size=model_input_size, mode="bilinear", align_corners=False)
    im_tensor = im_tensor / 255.0
    im_tensor = normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return im_tensor


def postprocess_mask(mask: torch.Tensor, im_size: tuple) -> torch.Tensor:
    """Resize a single mask back to original size, normalized to [0,255]."""
    H, W = im_size
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # (H,W) -> (1,1,H,W)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)  # (1,H,W) -> (1,1,H,W)
    
    mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask = (mask * 255).byte()
    return mask.squeeze(0)  # (1,H,W)


class BackgroundRemoval(nn.Module):
    def __init__(self, model_input_size=(1024, 1024)):
        super().__init__()
        self.model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-1.4", trust_remote_code=True
        )
        self.model_input_size = model_input_size

    @torch.no_grad()
    def forward(self, orig_images: np.ndarray, device=None):
        """
        Args:
            orig_images: np.ndarray (H,W,C) or (B,H,W,C)
        Returns:
            torch.Tensor (B,4,H,W) uint8 RGBA images
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if orig_images.ndim == 3:
            orig_images = orig_images[np.newaxis, ...]
        
        batch_size = orig_images.shape[0]
        orig_sizes = [img.shape[:2] for img in orig_images]

        inputs = preprocess_image(orig_images, self.model_input_size).to(device)
        
        # Model returns nested: result[0] is list of mask tensors
        result = self.model(inputs)[0]  # List of tensors, one per batch item

        rgba_batch = []
        for i in range(batch_size):
            # Get mask for this image and postprocess
            mask = postprocess_mask(result[i], orig_sizes[i]).cpu()  # (1,H,W)
            
            rgb = torch.tensor(orig_images[i], dtype=torch.uint8).permute(2, 0, 1)  # (3,H,W)
            rgba = torch.cat([rgb, mask], dim=0)  # (4,H,W)
            rgba_batch.append(rgba.unsqueeze(0))

        return torch.cat(rgba_batch, dim=0)  # (B,4,H,W)