import torch
from PIL import Image, ImageDraw

def crop_to_foreground(rgba_batch, padding=0):
    cropped_images = []

    for i in range(rgba_batch.shape[0]):
        rgba = rgba_batch[i]  # (4, H, W)
        alpha = rgba[3]  # (H, W)

        # Find non-zero alpha pixels
        nonzero_mask = alpha > 0
        nonzero_indices = torch.nonzero(nonzero_mask)

        if nonzero_indices.numel() == 0:
            # No foreground found, return empty or original
            cropped_images.append(rgba)
            continue

        # Get bounding box
        y_min = nonzero_indices[:, 0].min().item()
        y_max = nonzero_indices[:, 0].max().item()
        x_min = nonzero_indices[:, 1].min().item()
        x_max = nonzero_indices[:, 1].max().item()

        # Apply padding
        H, W = alpha.shape
        y_min = max(0, y_min - padding)
        y_max = min(H - 1, y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(W - 1, x_max + padding)

        # Crop
        cropped = rgba[:, y_min : y_max + 1, x_min : x_max + 1]
        cropped_images.append(cropped)

    return torch.stack(cropped_images)

def make_square_kornia(img_tensor):
    _, _, H, W = img_tensor.shape
    size = max(H, W)

    pad_h = size - H
    pad_w = size - W

    # Padding order: (left, right, top, bottom)
    padding = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]

    return torch.nn.functional.pad(img_tensor, padding, mode="constant", value=0)

def char2img(c):
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)

    # Measure bbox at origin
    left, top, right, bottom = draw.textbbox((0, 0), c, font=FONT)
    text_w, text_h = right - left, bottom - top

    # Center the glyph box
    x = (W - text_w) // 2
    y = (H - text_h) // 2

    # Shift by -left, -top so glyph is drawn correctly relative to bbox
    draw.text((x - left, y - top), c, fill=255, font=FONT)

    return img

def tensor_to_ascii(tensor):
    string = ""
    h, w = tensor.shape
    for i in range(h):
        for j in range(w):
            string += CHARS[tensor[i][j].item()]
        string += "\n"

    return string