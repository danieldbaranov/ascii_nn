import torch
from PIL import Image, ImageDraw

from torchvision.transforms import ToTensor


def char2img(c, font, w, h):
    """ Get image of character """
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
    """ Returns ascii art string from tensor"""
    string = ""
    h, w = tensor.shape
    for i in range(h):
        for j in range(w):
            string += charset[tensor[i][j].item()]
        string += "\n"

    return string

def _create_char_tensor(charset, font, w, h ):
        """ Creates character image tensors by calling char2img() """
        to_tensor = ToTensor()

        chars = []
        for c in charset:
            img = char2img(c, font, w, h)
            chars.append(to_tensor(img))

        chars = torch.stack(chars)
        return chars
