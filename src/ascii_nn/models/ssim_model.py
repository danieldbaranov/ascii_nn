import torch
import torch.nn as nn
import kornia
from ascii_nn.base_ascii_module import SimpleAsciiModule
from ascii_nn.charsets import SHIFT_JIS
from ascii_nn.utilities import _create_char_tensor


class SSIMModel(SimpleAsciiModule):
    def __init__(self, chars=SHIFT_JIS, target_rows=0, target_cols=0):
        super().__init__(target_rows=target_rows, target_cols=target_cols)
        self.chars = _create_char_tensor(chars, self.font, self.W, self.H)
        self.char_densities = self.chars.mean(dim=(1, 2, 3))

    def forward(self, img_tensor):
        img_tensor, num_rows, num_cols = self._prepare_image(img_tensor)

        img_tensor = kornia.filters.Canny()(img_tensor)[0]

        unfold = nn.Unfold(kernel_size=(self.H, self.W), stride=(self.H, self.W), padding=0)
        tiles = unfold(img_tensor).squeeze(0).T.view(-1, 1, self.H, self.W)

        num_tiles = len(tiles)
        num_refs = len(self.chars)

        # Expand for broadcasting
        tiles_exp = tiles[:, None].expand(-1, num_refs, -1, -1, -1).reshape(-1, 1, self.H, self.W)
        refs_exp = self.chars[None].expand(num_tiles, -1, -1, -1, -1).reshape(-1, 1, self.H, self.W)

        # Compute SSIM scores
        scores = kornia.metrics.ssim(tiles_exp, refs_exp, 3).mean(dim=(1, 2, 3))
        all_scores = scores.view(num_tiles, num_refs)

        # Density penalization
        tile_densities = tiles.mean(dim=(1, 2, 3))
        density_diff = (tile_densities[:, None] - self.char_densities[None, :]).abs()
        all_scores = all_scores - 3 * density_diff

        # Get best matches
        _, best_indices = all_scores.max(dim=1)

        return best_indices.view(num_rows, num_cols)
