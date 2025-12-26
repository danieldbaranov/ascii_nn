import torch
from torch import nn

from base_ascii_module import BaseAsciiModule

from constants import H, W

class Conv2DModel(BaseAsciiModule):
    def __init__(self, chars):
        super(Conv2DModel, self).__init__()
        self.glyph_kernels = self.normalize_kernels(chars)
    def normalize_kernels(self, k):
        k = k - k.mean(dim=(2,3), keepdim=True)
        k = k / (k.norm(dim=(2,3), keepdim=True) + 1e-6)
        return k
    def forward(self, img_tensor):
        img = img_tensor
        img = img - img.mean(dim=(2, 3), keepdim=True)
        img = img / (img.norm(dim=(2, 3), keepdim=True) + 1e-6)

        logits = nn.functional.conv2d(
            img,
            self.glyph_kernels,
            stride=(H, W)
        )

        print("logits shape", logits.shape)

        num_cols = img_tensor.shape[-1] // W

        # Get some of that confidence in there :0
        confidence, predictions = logits.max(dim=1)

        confident_mask = confidence >= 0.001

        final_predictions = torch.full_like(predictions, 0)

        final_predictions[confident_mask] = predictions[confident_mask] - 1

        return final_predictions.view(-1, num_cols)