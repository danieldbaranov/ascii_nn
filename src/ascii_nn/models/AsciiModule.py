import string

from bg_detection import BackgroundRemoval

from PIL import Image, ImageDraw, ImageFont
import kornia
from skimage import io
import torch
from torch import nn
from torchvision.transforms import ToTensor

from utilities import tensor_to_ascii, char2img, crop_to_foreground, make_square_kornia

CHARS = (" .,:;+*!?%#@$"
        + "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝﾞﾟ"
         + "|/\-_=~()[]{}<>") #+ "│┤╡╢╖╕╣║╗╝╜╛┐└┴┬├─┼╞╟╚╔╩╦╠═╬"

FONT_PATH = "./data/Saitamaar-Regular.ttf"
FONT_SIZE = 16
PAD = 0
RATIO = 2.5

FONT = ImageFont.truetype(FONT_PATH, FONT_SIZE)

H = FONT_SIZE + PAD
W = int((FONT_SIZE + PAD) / RATIO + 0.5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AsciiModule(nn.Module):
    def __init__(self, method="SSIM"):
        super().__init__()
        self.chars = self._create_char_tensor()
         # <-- add this
        #self.bg_remover = BackgroundRemoval().to(device)
        self.bg_remover = None
        self.edge_detection = kornia.filters.Canny(low_threshold=0.2, high_threshold=0.7)
        if method == "SSIM":
            self.model = SSIMModel(self.chars)
        elif method == "Conv2D":
            self.model = Conv2DModel(self.chars)
        else:
            raise NotImplementedError

    def _create_char_tensor(self):
        to_tensor = ToTensor()

        chars = []
        for c in CHARS:
            img = char2img(c)
            chars.append(to_tensor(img))


        chars = torch.stack(chars)
        return chars



    def preprocess(self, orig_im):
        if self.bg_remover is not None:
            # Background remover
            rgba_batch = self.bg_remover(orig_im, device=device)

            # Zoom on foreground
            rgba_batch = crop_to_foreground(rgba_batch, 32)

            # Make the image square
            rgba_batch = make_square_kornia(rgba_batch)

            # resize that shit to 512
            rgba_batch = kornia.geometry.transform.resize(
                rgba_batch, size=(512, 512), interpolation="nearest"  # or 'nearest'
            )
            alpha = rgba_batch[:, 3:].float() / 255.0
            rgb = rgba_batch[:, :3].float() / 255.0
            rgb = rgb * alpha + (1 - alpha)
        else:
            rgb = kornia.utils.image_to_tensor(orig_im)  / 255.0
            rgb = rgb.unsqueeze(0).float()

            print(rgb.shape)

        # Canny :)
        x_mag_bg, x_canny_bg = self.edge_detection(rgb)

        # Make that edge map thiccc
        img_tensor = x_canny_bg
        #img_tensor = kornia.morphology.dilation(x_canny_bg, torch.ones(3, 3))
        return img_tensor

    def forward(self, orig_im):
        img_tensor = self.preprocess(orig_im)
        return self.model(img_tensor)


if __name__ == "__main__":
    image_path = "../../tests/osaka.png"
    orig_im = io.imread(image_path)

    print("SSIM:")
    model = AsciiModule(method="SSIM")
    print(tensor_to_ascii(model(orig_im)))

    print("Conv2D:")
    model = AsciiModule(method="Conv2D")
    print(tensor_to_ascii(model(orig_im)))

