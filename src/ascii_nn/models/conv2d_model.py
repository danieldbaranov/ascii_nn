import torch
from torch import nn

from ascii_nn.base_ascii_module import SimpleAsciiModule
from ascii_nn.charsets import SHIFT_JIS
from ascii_nn.utilities import _create_char_tensor
import kornia


class Conv2DModel(SimpleAsciiModule):
    def __init__(self, chars=SHIFT_JIS, target_rows=0, target_cols=0):
        super().__init__(target_rows=target_rows, target_cols=target_cols)
        self.glyph_kernels = self._normalize_kernels(
            _create_char_tensor(chars, self.font, self.W, self.H)
        )

    def _normalize_kernels(self, k):
        k = k - k.mean(dim=(2, 3), keepdim=True)
        k = k / (k.norm(dim=(2, 3), keepdim=True) + 1e-6)
        return k

    def forward(self, img_tensor):
        img_tensor, num_rows, num_cols = self._prepare_image(img_tensor)

        img_tensor = kornia.filters.Canny()(img_tensor)[0]

        # Normalize input
        img = img_tensor - img_tensor.mean(dim=(2, 3), keepdim=True)
        img = img / (img.norm(dim=(2, 3), keepdim=True) + 1e-6)

        logits = nn.functional.conv2d(
            img,
            self.glyph_kernels,
            stride=(self.H, self.W)
        )

        # Get predictions with confidence thresholding
        confidence, predictions = logits.max(dim=1)
        confident_mask = confidence >= 0.001
        final_predictions = torch.where(confident_mask, predictions - 1, torch.zeros_like(predictions))

        return final_predictions.squeeze(0).view(num_rows, num_cols)
