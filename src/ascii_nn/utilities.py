import torch
from PIL import Image, ImageDraw

from torchvision.transforms import ToTensor

def make_square_kornia(img_tensor):
    _, _, H, W = img_tensor.shape
    size = max(H, W)

    pad_h = size - H
    pad_w = size - W

    # Padding order: (left, right, top, bottom)
    padding = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]

    return torch.nn.functional.pad(img_tensor, padding, mode="constant", value=0)

def char2img(c, font, w, h):
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)

    # Measure bbox at origin
    left, top, right, bottom = draw.textbbox((0, 0), c, font=font)
    text_w, text_h = right - left, bottom - top

    # Center the glyph box
    x = (w - text_w) // 2
    y = (h - text_h) // 2

    # Shift by -left, -top so glyph is drawn correctly relative to bbox
    draw.text((x - left, y - top), c, fill=255, font=font)

    return img

def tensor_to_ascii(tensor, charset):
    string = ""
    h, w = tensor.shape
    for i in range(h):
        for j in range(w):
            string += charset[tensor[i][j].item()]
        string += "\n"

    return string

def _create_char_tensor(charset, font, w, h ):
        to_tensor = ToTensor()

        chars = []
        for c in charset:
            img = char2img(c, font, w, h)
            chars.append(to_tensor(img))

        chars = torch.stack(chars)
        return chars