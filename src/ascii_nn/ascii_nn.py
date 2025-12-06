import numpy as np
import torch
import rembg

import cv2

from .ascii_model import load_model
from . import edge_detection

session = rembg.new_session()

# Charset: standard printable characters
#CHARS = string.printable[:-5]
#CHARS = " -_=|!/\\#*+@(){}[]<>%$"
#CHARS = " " + ''.join(chr(i) for i in range(0x2800, 0x2900))

EDGE_CHARS = r" -|+/\\_^v<>[]{}()~"
BG_CHARS = " .:-=+*#%@"
#BG_CHARS = " .·'`"

def get_square_crop(img_array):
    if img_array.shape[2] == 4:
        alpha = img_array[:, :, 3]
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)
    else:
        non_empty = np.any(img_array > 0, axis=2)
        rows = np.any(non_empty, axis=1)
        cols = np.any(non_empty, axis=0)

    if not rows.any() or not cols.any():
        return None

    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]

    width = right - left
    height = bottom - top
    size = max(width, height) + 1

    center_x = (left + right) // 2
    center_y = (top + bottom) // 2

    new_left = center_x - size // 2
    new_top = center_y - size // 2

    result = np.zeros((size, size, 4), dtype=img_array.dtype)

    src_left = max(0, new_left)
    src_top = max(0, new_top)
    src_right = min(img_array.shape[1], new_left + size)
    src_bottom = min(img_array.shape[0], new_top + size)

    dst_left = src_left - new_left
    dst_top = src_top - new_top
    dst_right = dst_left + (src_right - src_left)
    dst_bottom = dst_top + (src_bottom - src_top)

    result[dst_top:dst_bottom, dst_left:dst_right] = img_array[src_top:src_bottom, src_left:src_right]

    return result

class ascii_nn:
    def __init__(self):

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Metal (MPS) acceleration.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA acceleration.")
        else:
            self.device = torch.device("cpu")
            print("Using CPU.")
        self.edge_model, self.bg_model, self.tile_h, self.tile_w = load_model()

    def convert(self, source_img, n_lines=40, noise_reduction=2, line_weight=1, text_ratio=2.5):
        """Main entry point to convert image to ASCII."""

        # remove background
        height, width = source_img.shape[:2]
        source_img = cv2.resize(source_img, (width * 4, height * 4), interpolation=cv2.INTER_CUBIC)
        source_img = rembg.remove(source_img, session=session)

         # Get bounding box of non-transparent pixels
        source_img = get_square_crop(source_img)

        # Get Edges
        edges, rows, cols = edge_detection.get_edges(
            source_img, n_lines,
            sigma=noise_reduction,
            weight=line_weight,
            ratio=text_ratio,
            tile_h=self.tile_h, tile_w=self.tile_w
        )

        background, rows, cols = edge_detection.get_raw_image(
            source_img, n_lines,
            ratio=text_ratio,
            tile_h=self.tile_h,
            tile_w=self.tile_w
        )

        #Image.fromarray((edges * 255).astype(np.uint8)).save("test.png")


        # Slice into tiles
        # We want shape: (N_TILES, 1, H, W) for PyTorch
        edge_tiles = []
        bg_tiles = []
        for r in range(rows):
            for c in range(cols):
                y_start = r * self.tile_h
                x_start = c * self.tile_w
                # Extract tile
                edge_tiles.append(edges[y_start : y_start+self.tile_h, x_start : x_start+self.tile_w])
                bg_tiles.append(background[y_start : y_start+self.tile_h, x_start : x_start+self.tile_w])

        # Stack into a batch tensor
        edge_batch_np = np.array(edge_tiles)
        bg_batch_np = np.array(bg_tiles)
        edge_batch_tensor = torch.tensor(edge_batch_np, dtype=torch.float32).unsqueeze(1).to(self.device) # Add channel dim
        bg_batch_tensor = torch.tensor(bg_batch_np, dtype=torch.float32).unsqueeze(1).to(self.device) # Add channel dim


        # Inference in batches to avoid OOM on large images
        edge_preds = []
        bg_preds = []
        inference_batch_size = 512

        with torch.no_grad():
            for i in range(0, len(edge_batch_tensor), inference_batch_size):
                batch = edge_batch_tensor[i : i + inference_batch_size]
                outputs = self.edge_model(batch)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                edge_preds.extend(preds)

            for i in range(0, len(bg_batch_tensor), inference_batch_size):
                batch = bg_batch_tensor[i : i + inference_batch_size]
                outputs = self.bg_model(batch)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                bg_preds.extend(preds)

        # Reconstruct String
        ascii_str = ""
        for r in range(rows):
            line = ""
            for c in range(cols):
                idx = r * cols + c
                char_idx = edge_preds[idx]
                char = EDGE_CHARS[char_idx]
                if char == " ":
                    char = BG_CHARS[bg_preds[idx]]
                line += char
            ascii_str += line + "\n"

        return ascii_str


