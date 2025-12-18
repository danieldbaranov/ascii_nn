import string

from bg_detection import BackgroundRemoval

from PIL import Image, ImageDraw, ImageFont
import kornia
from skimage import io
import torch
from torch import nn
from torchvision.transforms import ToTensor

def chars(start, end):
    return ''.join(chr(c) for c in range(start, end + 1))

def get_shift_jis_chars(encoding='cp932'):
    all_chars_unicode = ''.join(chr(i) for i in range(0x110000))
    supported_chars_bytes = all_chars_unicode.encode(encoding, errors='ignore')
    supported_chars = supported_chars_bytes.decode(encoding)
    return supported_chars

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

class SSIMModel(nn.Module):
    def __init__(self, chars):
        super(SSIMModel, self).__init__()
        self.chars = chars
        self.char_densities = self.chars.mean(dim=(1, 2, 3))

    def forward(self, img_tensor):
        unfold = nn.Unfold(kernel_size=(H, W), stride=(H, W), padding=0)

        # threshold for those hard edges
        # img_tensor = img_tensor.clamp(0, 1).round()

        # called horse bc that was the og image
        horse = unfold(img_tensor).squeeze(0).T.view(-1, 1, H, W)

        num_tiles = len(horse)
        num_refs = len(self.chars)
        all_scores = torch.zeros(num_tiles, num_refs)

        chunk = horse  # (chunk_size, 1, H, W)
        cs = chunk.shape[0]

        # Expand for broadcasting
        chunk_exp = chunk[:, None].expand(-1, num_refs, -1, -1, -1).reshape(-1, 1, H, W)
        refs_exp = self.chars[None].expand(cs, -1, -1, -1, -1).reshape(-1, 1, H, W)

        scores = kornia.metrics.ssim(chunk_exp, refs_exp, 3).mean(dim=(1, 2, 3))
        all_scores = scores.view(cs, num_refs)

        # Density penalization
        tile_densities = chunk.mean(dim=(1, 2, 3))  # (num_tiles,)
        density_diff = (tile_densities[:, None] - self.char_densities[None, :]).abs()

        # Combine: SSIM rewards shape match, penalty for density mismatch
        all_scores = all_scores - 3 * density_diff  # tune this weight

        # Now build your string
        best_scores, best_indices = all_scores.max(dim=1)
        num_cols = img_tensor.shape[-1] // W

        return best_indices.view(-1, num_cols)

class Conv2DModel(nn.Module):
    def __init__(self, chars):
        super(Conv2DModel, self).__init__()
        self.glyph_kernels = self.normalize_kernels(chars)
    def normalize_kernels(self, k):
        k = k - k.mean(dim=(2,3), keepdim=True)
        k = k / (k.norm(dim=(2,3), keepdim=True) + 1e-6)
        return k
    def forward(self, img_tensor):
        img = img_tensor
        img = img - img.mean(dim=(2, 3), keepdim=True)
        img = img / (img.norm(dim=(2, 3), keepdim=True) + 1e-6)

        logits = nn.functional.conv2d(
            img,
            self.glyph_kernels,
            stride=(H, W)
        )

        print("logits shape", logits.shape)

        num_cols = img_tensor.shape[-1] // W

        # Get some of that confidence in there :0
        confidence, predictions = logits.max(dim=1)

        confident_mask = confidence >= 0.001

        final_predictions = torch.full_like(predictions, 0)

        final_predictions[confident_mask] = predictions[confident_mask] - 1

        return final_predictions.view(-1, num_cols)

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

def tensor_to_ascii(tensor):
    string = ""
    h, w = tensor.shape
    for i in range(h):
        for j in range(w):
            string += CHARS[tensor[i][j].item()]
        string += "\n"

    return string

if __name__ == "__main__":
    image_path = "../../tests/osaka.png"
    orig_im = io.imread(image_path)

    print("SSIM:")
    model = AsciiModule(method="SSIM")
    print(tensor_to_ascii(model(orig_im)))

    print("Conv2D:")
    model = AsciiModule(method="Conv2D")
    print(tensor_to_ascii(model(orig_im)))