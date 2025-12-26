from abc import ABC, abstractmethod
from torch import nn
from .constants import H, W

class SimpleAsciiModule(nn.Module, ABC):
    @abstractmethod
    def __init__(self, preprocess: callable=None, target_rows=None, target_cols=None):
        super().__init__()
        self.preprocess = preprocess if preprocess is not None else lambda x: x
        # Store target dimensions - these are set in __init__ to avoid if statements in forward
        self.target_rows = target_rows
        self.target_cols = target_cols
        # Determine dimension calculation mode at init time
        self.use_target_rows = target_rows is not None
        self.use_target_cols = target_cols is not None

    @abstractmethod
    def info(self):
        pass

    @abstractmethod
    def forward(self, orig_im):
        img_tensor = self.preprocess(orig_im)
        pass
    
    def _calculate_output_dims(self, img_tensor):
        """Calculate output dimensions based on stored target values or image size."""
        # Use stored target values if available, otherwise calculate from image
        # Avoids if statements by using list indexing with boolean-to-int conversion
        img_rows = img_tensor.shape[-2] // H
        img_cols = img_tensor.shape[-1] // W
        
        # Use list indexing: [img_rows, self.target_rows][int(self.use_target_rows)]
        # This avoids if statements and is fully compilable
        num_rows = [img_rows, self.target_rows][int(self.use_target_rows)]
        num_cols = [img_cols, self.target_cols][int(self.use_target_cols)]
        
        return num_rows, num_cols

class TrainableAsciiModule(SimpleAsciiModule):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def info(self):
        pass

    @abstractmethod
    def forward(self, orig_im):
        img_tensor = self.preprocess(orig_im)
        pass