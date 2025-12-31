import pytest
import kornia
from ascii_nn.models import Conv2DModel, SSIMModel, IntensityModel
from ascii_nn.utilities import tensor_to_ascii
from ascii_nn.charsets import SHIFT_JIS

MODEL_CLASSES = [Conv2DModel, SSIMModel, IntensityModel]
TEST_IMAGES = ["tests/osaka.png", "tests/horse.png", "tests/boat.png"]
SIZE_CONFIGS = [
    {"target_rows": 20, "target_cols": 0},   # rows only (small)
    {"target_rows": 50, "target_cols": 0},   # rows only (medium)
    {"target_rows": 80, "target_cols": 0},   # rows only (large)
    {"target_rows": 0, "target_cols": 40},   # cols only (small)
    {"target_rows": 0, "target_cols": 80},   # cols only (medium)
    {"target_rows": 0, "target_cols": 120},  # cols only (large)
    {"target_rows": 30, "target_cols": 60},  # both
    {"target_rows": 50, "target_cols": 100}, # both (larger)
]


def load_image(path):
    """Load and preprocess image to grayscale tensor."""
    img = kornia.io.load_image(path, kornia.io.ImageLoadType.RGB32)
    if img.shape[0] == 3:
        img = kornia.color.rgb_to_grayscale(img)
    return img.unsqueeze(0).float()


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=lambda m: m.__name__)
@pytest.mark.parametrize("size_config", SIZE_CONFIGS, ids=lambda c: f"r{c['target_rows']}_c{c['target_cols']}")
@pytest.mark.parametrize("image_path", TEST_IMAGES, ids=lambda p: p.split("/")[-1].replace(".png", ""))
def test_size_constraints(model_class, size_config, image_path):
    """Test model output dimensions match constraints."""
    model = model_class(**size_config)
    output = model(load_image(image_path))
    
    assert output.dim() == 2
    if size_config["target_rows"]:
        assert output.shape[0] == size_config["target_rows"]
    if size_config["target_cols"]:
        assert output.shape[1] == size_config["target_cols"]
    assert output.min() >= 0


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=lambda m: m.__name__)
def test_ascii_output(model_class):
    """Test ASCII conversion produces expected output."""
    output = model_class(target_rows=30)(load_image("tests/osaka.png"))
    ascii_art = tensor_to_ascii(output, SHIFT_JIS)
    
    assert isinstance(ascii_art, str) and len(ascii_art) > 0
    assert len(ascii_art.strip().split("\n")) == 30


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=lambda m: m.__name__)
def test_no_constraints(model_class):
    """Test model derives size from image when no constraints given."""
    output = model_class()(load_image("tests/osaka.png"))
    
    assert output.dim() == 2
    assert output.shape[0] > 0 and output.shape[1] > 0
