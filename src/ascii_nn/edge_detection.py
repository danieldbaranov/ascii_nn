import os
import numpy as np

import numpy as np
import cv2
import os

import rembg

session = rembg.new_session()


def get_edges(img_bgr, n_lines, sigma=1.5, weight=1.0, ratio=2.5, tile_h=35, tile_w=14,
              low_threshold=30, high_threshold=80):
    """
    Detects edges using Canny edge detection and prepares the grid.
    
    Args:
        img_bgr: BGR numpy array input
        n_lines: Number of output lines
        sigma: Gaussian blur sigma before edge detection
        weight: Controls edge thickness - higher = thicker lines
        ratio: Aspect ratio adjustment
        tile_h: Tile height in pixels
        tile_w: Tile width in pixels
        low_threshold: Canny low threshold (0-255)
        high_threshold: Canny high threshold (0-255)
    
    Returns:
        edges_final: Edge map as float32 array (0-1)
        n_lines: Number of lines
        width_chars: Width in characters
    """
    # Calculate dimensions
    im_h, im_w = img_bgr.shape[:2]
    width_chars = int(im_w / im_h * n_lines * ratio + 0.5)
    
    # Target size in pixels
    target_h = tile_h * n_lines
    target_w = tile_w * width_chars
    
    # Resize for processing
    img_resized = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur to reduce noise
    if sigma > 0:
        kernel_size = int(sigma * 4) | 1  # ensure odd
        gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    
    # Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # Convert to 0-1 float
    edges_final = (edges / 255.0).astype(np.float32)
    
    # Apply weight for line thickness
    if weight != 1.0:
        kernel_size = max(1, int(abs(weight - 1.0) * 3) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if weight > 1.0:
            edges_final = cv2.dilate(edges_final, kernel, iterations=1)
        else:
            edges_final = cv2.erode(edges_final, kernel, iterations=1)
    
    return edges_final, n_lines, width_chars

def get_raw_image(img_bgr, n_lines, ratio=2.5, tile_h=35, tile_w=14):
    # Convert to grayscale numpy array
    #img_gray = np.array(img_pil.convert("L"))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Calculate dimensions
    im_h, im_w = img_gray.shape[:2]
    width_chars = int(im_w / im_h * n_lines * ratio + 0.5)
    
    # Target size in pixels
    target_h = tile_h * n_lines
    target_w = tile_w * width_chars
    
    # Resize to target dimensions
    img_resized = cv2.resize(img_gray, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    #img_resized = rembg.remove(img_resized, bgcolor=(255, 255, 255, 255), session=session)


    img_resized = cv2.equalizeHist(img_resized)
    
    # Normalize to 0-1 float32
    img_final = img_resized.astype(np.float32) / 255.0
    
    return img_final, n_lines, width_chars   