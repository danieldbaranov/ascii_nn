import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from platformdirs import user_cache_dir

FONT_SIZE = 35
FONT_PATH = "/System/Library/Fonts/Apple Symbols.ttf"

EDGE_CHARS = r" -|+/\\_^v<>[]{}()~"
BG_CHARS = " .:-=+*#%@"
#BG_CHARS = " .·'`"

PAD = 0
BATCH_SIZE = 64
EPOCHS = 10

model_path = os.path.join(user_cache_dir("ascii_nn"), "models")
os.makedirs(model_path, exist_ok=True)
EDGE_MODEL_PATH = os.path.join(model_path, "edge_ascii_model.pth")
BG_MODEL_PATH = os.path.join(model_path, "bg_ascii_model.pth")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    #print(f"Using CUDA acceleration.")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    #print(f"Using Apple Metal (MPS) acceleration.")
else:
    DEVICE = torch.device("cpu")
    #print(f"Using CPU.")

def salt_pepper_noise(img, amount=0.05):
    noise = torch.rand_like(img)
    img = img.clone()
    img[noise < amount/2] = 0
    img[noise > 1 - amount/2] = 1
    return img

class CharDataset(Dataset):
    """Generates augmented character images on the fly."""
    def __init__(self, font_path, font_size, chars, epoch_length=1000):
        self.font = ImageFont.truetype(font_path, font_size)
        self.chars = chars
        self.char_map = {c: i for i, c in enumerate(chars)}
        self.epoch_length = epoch_length # Virtual length for the epoch

        self.ratio = 2.5
        self.h = font_size + PAD
        self.w = int((font_size + PAD) / self.ratio + 0.5)

        self.base_imgs = {c: self._char2img(c) for c in chars}

        self.transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=10,
                translate=(0.15, 0.15),
                scale=(0.8, 1.2),
                shear=(-8, 8),
                fill=0
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.RandomApply([
                transforms.Lambda(lambda x: torch.pow(x, torch.empty(1).uniform_(0.7, 1.4).item()))
            ], p=0.3),
            transforms.RandomApply([
                transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x))
            ], p=0.5),
            transforms.RandomApply([
                transforms.Lambda(lambda x: salt_pepper_noise(x, amount=0.03))
            ], p=0.2),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.0), value=0),
            transforms.Lambda(lambda x: x.clamp(0, 1))
        ])
    def _char2img(self, c):
        img = Image.new("L", (self.w, self.h), 0) # Grayscale (L)
        draw = ImageDraw.Draw(img)
        # Use simple centering logic
        # Note: textbbox is newer, but textsize is deprecated. 
        # Using bounding box for centering:
        left, top, right, bottom = draw.textbbox((0,0), c, font=self.font)
        text_w, text_h = right - left, bottom - top
        
        x = (self.w - text_w) // 2
        y = (self.h - text_h) // 2
        # Adjust for baseline offset if needed, but simple centering usually works for ASCII
        draw.text((x, y - top/2), c, fill=255, font=self.font)
        return img

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        # Randomly select a character
        char = self.chars[np.random.randint(0, len(self.chars))]
        label = self.char_map[char]
        
        img = self.base_imgs[char]
        
        # Apply augmentation
        img_tensor = self.transform(img)
        
        return img_tensor, label


class CNN(nn.Module):
    def __init__(self, num_classes, h, w):
        super(CNN, self).__init__()
        # Conv2D: (in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1) 
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        
        # Calculate size after flattening
        self.fc_input_dim = 32 * h * w 
        self.fc = nn.Linear(self.fc_input_dim, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x

def train_model(pallete, path):
    print("Generating training data and training model...")
    dataset = CharDataset(FONT_PATH, FONT_SIZE, pallete, epoch_length=BATCH_SIZE * 128)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CNN(len(pallete), dataset.h, dataset.w).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(loader):.4f} - Acc: {100 * correct / total:.2f}%")

    torch.save({
        'model_state_dict': model.state_dict(),
        'chars': pallete,
        'h': dataset.h,
        'w': dataset.w
    }, path)
    print(f"Model saved to {path}")
    return model, dataset.h, dataset.w

def load_model():
    if not os.path.exists(EDGE_MODEL_PATH) or not os.path.exists(BG_MODEL_PATH):
        train_model(EDGE_CHARS, EDGE_MODEL_PATH)
        train_model(BG_CHARS, BG_MODEL_PATH)
        return load_model()

    print(f"Loading model from {EDGE_MODEL_PATH}...")
    edge_checkpoint = torch.load(EDGE_MODEL_PATH, map_location=DEVICE, weights_only=True)
    h, w = edge_checkpoint['h'], edge_checkpoint['w']
    edge_chars = edge_checkpoint['chars']
    edge_model = CNN(len(edge_chars), h, w).to(DEVICE)
    edge_model.load_state_dict(edge_checkpoint['model_state_dict'])
    edge_model.eval()

    print(f"Loading model from {BG_MODEL_PATH}...")
    bg_checkpoint = torch.load(BG_MODEL_PATH, map_location=DEVICE, weights_only=True)
    h, w = bg_checkpoint['h'], bg_checkpoint['w']
    bg_chars = bg_checkpoint['chars'] 
    bg_model = CNN(len(bg_chars), h, w).to(DEVICE)
    bg_model.load_state_dict(bg_checkpoint['model_state_dict'])
    bg_model.eval()
    
    return edge_model, bg_model, h, w