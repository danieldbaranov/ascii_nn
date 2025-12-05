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

    
    def convert(self, img_path, n_lines=40, noise_reduction=2, line_weight=1, text_ratio=2.5):
        """Main entry point to convert image to ASCII."""

        # Load Image
        try:
            source_img = cv2.imread(img_path)
        except Exception as e:
            print(f"Could not open image: {e}")
            return

        # remove background
        height, width = source_img.shape[:2]
        source_img = cv2.resize(source_img, (width * 4, height * 4), interpolation=cv2.INTER_CUBIC)
        source_img = rembg.remove(source_img, session=session)
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGRA2BGR)
        print(source_img.shape)

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


