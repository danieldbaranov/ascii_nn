from abc import ABC, abstractmethod
from torch import nn
import torch.nn.functional as F
from PIL import ImageFont
import importlib.resources

FONT_SIZE = 16
PAD = 0
RATIO = 2.5

FONT_PATH = str(importlib.resources.files('ascii_nn.data') / 'Saitamaar-Regular.ttf')
FONT = ImageFont.truetype(FONT_PATH, FONT_SIZE)

H = FONT_SIZE + PAD
W = int((FONT_SIZE + PAD) / RATIO + 0.5)


class SimpleAsciiModule(nn.Module, ABC):
    """
    Base class for ASCII art generation modules.
    
    Args:
        preprocess: Optional preprocessing function for input tensors
        target_rows: Fixed output rows (0 = derive from image/aspect ratio)
        target_cols: Fixed output cols (0 = derive from image/aspect ratio)
        font: PIL font for character rendering
        font_size: Font size in pixels
    """
    
    def __init__(self, preprocess=None, target_rows=0, target_cols=0, font=FONT, font_size=FONT_SIZE):
        super().__init__()
        self.target_rows = target_rows
        self.target_cols = target_cols
        self.font = font
        self.H = font_size + PAD
        self.W = int((font_size + PAD) / RATIO + 0.5)

    def _prepare_image(self, img_tensor):
        """Resize image preserving aspect ratio based on target dimensions."""
        img_h, img_w = img_tensor.shape[-2], img_tensor.shape[-1]
        aspect = img_w / img_h  # pixel aspect ratio
        
        if self.target_rows and self.target_cols:
            num_rows, num_cols = self.target_rows, self.target_cols
        elif self.target_rows:
            num_rows = self.target_rows
            num_cols = max(1, round(self.target_rows * aspect * self.H / self.W))
        elif self.target_cols:
            num_cols = self.target_cols
            num_rows = max(1, round(self.target_cols / aspect * self.W / self.H))
        else:
            num_rows = img_h // self.H
            num_cols = img_w // self.W
        
        target_size = (num_rows * self.H, num_cols * self.W)
        img_tensor = F.interpolate(img_tensor, size=target_size, mode='bilinear', align_corners=False)
        
        return img_tensor, num_rows, num_cols

    @abstractmethod
    def forward(self, img_tensor):
        pass