from abc import ABC, abstractmethod
from torch import nn
import torch.nn.functional as F
from PIL import ImageFont
import importlib.resources

FONT_SIZE = 16
PAD = 0
RATIO = 2.5


FONT_PATH = str(importlib.resources.files('ascii_nn.data') / 'SourceHanCodeJP.ttc')
FONT = ImageFont.truetype(FONT_PATH, FONT_SIZE)

H = FONT_SIZE + PAD
W = int((FONT_SIZE + PAD) / RATIO + 0.5)
