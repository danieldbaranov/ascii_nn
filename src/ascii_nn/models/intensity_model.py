import torch
from torch import nn

from ascii_nn.base_ascii_module import SimpleAsciiModule
from ascii_nn.charsets import SHIFT_JIS
from ascii_nn.utilities import _create_char_tensor


class IntensityModel(SimpleAsciiModule):
    """ Tone-based ASCII art model using intensity matching """
    
    def __init__(self, chars=SHIFT_JIS, target_rows=0, target_cols=0):
        super().__init__(target_rows=target_rows, target_cols=target_cols)
        
        # Create character tensor
        self.char_tensor = _create_char_tensor(chars, self.font, self.W, self.H)
        
        # Calculate intensity (mean brightness) for each character
        self.char_intensities = self.char_tensor.mean(dim=(1, 2, 3))
        
        # Sort characters by intensity
        self.sorted_indices = torch.argsort(self.char_intensities)
        self.sorted_intensities = self.char_intensities[self.sorted_indices]

    def forward(self, img_tensor):
        img_tensor, num_rows, num_cols = self._prepare_image(img_tensor)
        
        # Unfold image into tiles
        unfold = nn.Unfold(kernel_size=(self.H, self.W), stride=(self.H, self.W), padding=0)
        tiles = unfold(img_tensor).squeeze(0).T  # Shape: (num_tiles, H*W)
        
        # Calculate tile intensities
        tile_intensities = tiles.mean(dim=1)
        
        # Normalize tile intensities to span the full character intensity range
        tile_min = tile_intensities.min()
        tile_max = tile_intensities.max()
        char_min = self.char_intensities.min()
        char_max = self.char_intensities.max()
        
        # Stretch tile intensities from [tile_min, tile_max] to [char_min, char_max]
        if tile_max - tile_min > 1e-6:  # Avoid division by zero
            normalized_tiles = (tile_intensities - tile_min) / (tile_max - tile_min)
            normalized_tiles = normalized_tiles * (char_max - char_min) + char_min
        else:
            normalized_tiles = tile_intensities
        
        # Invert intensities
        intensity_diff = ((char_max + char_min - normalized_tiles[:, None]) - self.char_intensities[None, :]).abs()
        
        # Get index of character with minimum intensity difference
        best_indices = intensity_diff.argmin(dim=1)
        
        return best_indices.view(num_rows, num_cols)

