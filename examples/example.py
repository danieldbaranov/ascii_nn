from ascii_nn.models import SSIMModel, IntensityModel
from ascii_nn.utilities import tensor_to_ascii
from ascii_nn.charsets import SHIFT_JIS, BLOCK_SHADING

import kornia

CHARSET = SHIFT_JIS

PATH = "../tests/images/horse.png"

def load_image(path):
    """Load and preprocess image to grayscale tensor."""
    img = kornia.io.load_image(path, kornia.io.ImageLoadType.RGB32)
    if img.shape[0] == 3:
        img = kornia.color.rgb_to_grayscale(img)
    return img.unsqueeze(0).float()

if __name__ == "__main__":

    print("Intensity w/ Block Shading")
    CHARSET = BLOCK_SHADING
    model = IntensityModel(target_rows=24, chars=CHARSET)
    img = load_image(PATH)
    print(tensor_to_ascii(model(img), charset=CHARSET))

    print("SSIM w/ SHIFT JIS")
    CHARSET = SHIFT_JIS
    model = SSIMModel(target_rows=24, chars=CHARSET)
    print(tensor_to_ascii(model(img), charset=CHARSET))