IMAGE_SIZE=128
MAX_IMAGES_TO_LOAD=2000000
LATENT_DIM = 2048
AUTOENCODER_NOISE_FACTOR = 0.05
NUM_DATASET_WORKERS = 0


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
import sys
import os
import numpy as np
import cv2


from PIL import Image
import random

device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    try:
        device = torch.device("mps")
    except:
        device = torch.device("cpu")


def add_positional_encoding(tensor):
    # Extract dimensions
    B, C, W, H = tensor.shape

    # Create positional encodings
    pos_x = torch.linspace(-1, 1, W).view(1, 1, W, 1).expand(B, 1, W, H).to(tensor.device)
    pos_y = torch.linspace(-1, 1, H).view(1, 1, 1, H).expand(B, 1, W, H).to(tensor.device)

    # Concatenate positional encodings along the channel dimension
    pos_encodings = torch.cat([pos_x, pos_y], dim=1)
    tensor_with_pos = torch.cat([tensor, pos_encodings], dim=1)
    return tensor_with_pos

def save_tensor_image(tensor, filename):
    tensor = tensor.detach().cpu()
    tensor = tensor.clamp(0, 1)
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.numpy()
    tensor = (tensor * 255).astype(np.uint8)
    image = Image.fromarray(tensor)
    image.save(filename)

def rgb_to_stress(image, colorbar):
    """Converts RGB values to stress based on a colorbar.

    Args:
        image: A PyTorch tensor (Batch, 3, width, height) with RGB values.
        colorbar: A numpy array (64, 3) representing the colorbar.

    Returns:
        A PyTorch tensor (Batch, 1, width, height) with stress values.
    """

    # Reshape the colorbar and image for broadcasting
    #for i in range(len(colorbar)):
     #   color = colorbar[i]
     #   image[image == color] = i/65.0
    #return image


    colorbar = torch.tensor(colorbar)#.to(image.device) # Convert to PyTorch tensor
    colorbar = colorbar.view(1, 65, 3, 1, 1)  # Add dimensions for broadcasting
    image = image.view(image.shape[0], 1, 3, image.shape[2], image.shape[3])

    # Calculate distances between each pixel and each color in the colorbar
    distances = (image - colorbar).pow(2).sum(dim=2)

    # Find the index of the closest color in the colorbar for each pixel
    indices = distances.argmin(dim=1)

    # Convert the indices to linear stress values (0 to 1)
    stress_values = indices.float() / (colorbar.shape[1] - 1)

    return stress_values.unsqueeze(1)  # Add channel dimension for stress

def stress_to_rgb(stress_values, colorbar):
    """Converts stress values to RGB based on a colorbar.

    Args:
        stress_values: A PyTorch tensor (Batch, 1, width, height) with stress values (0-1).
        colorbar: A numpy array (64, 3) representing the RGB colorbar (0-1).

    Returns:
        A PyTorch tensor (Batch, 3, width, height) with RGB values (0-1).
    """
    colorbar = torch.tensor(colorbar).to(stress_values.device) # Convert to PyTorch tensor

    # Ensure values are within valid range
    stress_values = stress_values.clamp(0, 1)

    # Convert stress values to colorbar indices
    indices = (stress_values * (colorbar.shape[0] - 1)).long()

    # Reshape the colorbar and image for broadcasting
    colorbar = torch.tensor(colorbar).to(stress_values.device)
    colorbar = colorbar.view(1, colorbar.shape[0], 3, 1, 1)

    # Convert the indices to RGB values using the colorbar
    rgb_image = colorbar[0, indices]

    return rgb_image

class LatentTranslator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, num_layers=3, hidden_dim=LATENT_DIM):
        super(LatentTranslator, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(latent_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, latent_dim))
        self.layers.append(nn.ReLU())
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PairedImageDataset(Dataset):
    def __init__(self, image_dir, colorbar, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        if (transform is None):
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
                torchvision.transforms.ToTensor(),
            ])
        self.image_pairs = self._find_pairs()
        self.colorbar = colorbar.cpu()

    def _find_pairs(self):
      image_pairs = [(filename, filename[:-8] + "_OUT.png")
                for filename in os.listdir(self.image_dir)
                if filename.endswith("_INP.png")][:MAX_IMAGES_TO_LOAD]
      return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        inp_name, out_name = self.image_pairs[idx]
        inp_path = os.path.join(self.image_dir, inp_name)
        out_path = os.path.join(self.image_dir, out_name)

        inp_image = Image.open(inp_path)
        out_image = Image.open(out_path)

        flip_transform = torchvision.transforms.RandomHorizontalFlip(p=1.0)
        if random.random() < 0.5:
            inp_image = flip_transform(inp_image)
            out_image = flip_transform(out_image)


        if self.transform:
            inp_image = self.transform(inp_image)
            out_image = self.transform(out_image)
      #  stress_image = rgb_to_stress(out_image.unsqueeze(0), self.colorbar).float()

       # normalized_stress_image = stress_image - stress_image.min()
       # normalized_stress_image /= normalized_stress_image.abs().max()
       # out_image = normalized_stress_image.squeeze(0)

       # out_image = stress_to_rgb(normalized_stress_image, self.colorbar).squeeze(-1).squeeze(0).squeeze(-1).float()
        #out_image = out_image.squeeze(0).permute(2,0,1)

        #out_image = (out_image * 2) - 1

        return inp_image, out_image

def load_color_bar(color_bar_path):
    # Load color bar
    color_bar_img = cv2.imread(color_bar_path)
    color_bar_img = cv2.cvtColor(color_bar_img, cv2.COLOR_BGR2RGB)
    color_bar_img = color_bar_img.astype(float) / 255.0
    colors = [np.array([1.0, 1.0, 1.0])]
    prev_color = np.array([0, 0, 0])

    for col in color_bar_img[0]:
        if not np.allclose(col, prev_color):
            colors.append(col)
            prev_color = col

    colors = torch.tensor(colors, dtype=torch.float32).to(device)
    return colors

def generate_color_weights_torch(image, colors):
    # Generate weights
    weights = torch.linspace(1, 10, len(colors)).to(device)

    # Convert image to float and normalize
    image = image.float() #/ 255.0
    batch_size, channels, height, width = image.shape

    reshaped_image = image.permute(0, 2, 3, 1).reshape(-1, 3)

    # Calculate distances between each pixel and colors
    distances = torch.cdist(reshaped_image.unsqueeze(0), colors.unsqueeze(0))

    # Find closest colors
    closest_color_idx = torch.argmin(distances, dim=2).squeeze()

    # Map weights and reshape to original image shape
    image_weights = weights[closest_color_idx].view(batch_size, height, width)

    return image_weights

def stress_pattern_loss(output, target, mask):
    # Calculate loss only on stress pattern regions
    pattern_loss = F.mse_loss(output * mask, target * mask, reduction='mean')
    return pattern_loss

class ConvAutoencoder(nn.Module):
    def __init__(self, channels, channel_sizes=[16, 32, 64, 128, 256], latent_dim=LATENT_DIM):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        encoder_layers = []
        in_channels = channels
        for out_channels in channel_sizes:
            encoder_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            encoder_layers.append(nn.BatchNorm2d(out_channels))
            encoder_layers.append(nn.ReLU(True))
            encoder_layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder_layers)

        # Bottleneck (fully connected layers for latent representation)
        self.fc1 = nn.Linear(channel_sizes[-1] * 4 * 4, latent_dim)  # Assuming downsampled to 4x4
        self.fc2 = nn.Linear(latent_dim, channel_sizes[-1] * 4 * 4)

        # Decoder (mirror the encoder structure)
        decoder_layers = []
        for i in range(len(channel_sizes) - 1, 0, -1):
            decoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            decoder_layers.append(nn.Conv2d(channel_sizes[i], channel_sizes[i - 1], kernel_size=3, padding=1))
            decoder_layers.append(nn.BatchNorm2d(channel_sizes[i - 1]))
            decoder_layers.append(nn.ReLU(True))
        decoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        decoder_layers.append(
            nn.Conv2d(channel_sizes[0], 3, kernel_size=3, padding=1)
        )
       # decoder_layers.append(nn.Conv2d(3,3, kernel_size=3, padding=1))
        decoder_layers.append(nn.Conv2d(3,128, kernel_size=1))
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Conv2d(128,3, kernel_size=1))
        self.decoder = nn.Sequential(*decoder_layers)

    def to_latent(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x
    def from_latent(self, x):
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), -1, 4, 4)
        x = self.decoder(x)
        return x

    def forward(self, x):
        latent = self.to_latent(x)
        latent = latent + torch.randn_like(latent) * AUTOENCODER_NOISE_FACTOR
        x = self.from_latent(latent)
        return x, latent

colors = load_color_bar("dual_autoencoders/color_bar.png")

model1 = ConvAutoencoder(5).to(device)
model2 = ConvAutoencoder(5).to(device)
latent_translator = LatentTranslator().to(device)

### INPUT: ARG 1 ###
### path to dataset, includes INP and OUT images ###
dataset = sys.argv[1]
dataset = PairedImageDataset(dataset, colors)
#dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
print(len(dataset))

### OUTPUT: ARG 2 ###
### path to save images and models ###
save_path = sys.argv[2]

### PARAMS ###
### episodes, start_lr, end_lr ###
num_epochs = 1000

optimizer = torch.optim.AdamW(list(model1.parameters()) + list(model2.parameters()) + list(latent_translator.parameters()), lr=1e-3)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=NUM_DATASET_WORKERS, pin_memory=True)
model1.train()
model2.train()
latent_translator.train()
i = 0
for epoch in range(num_epochs):
    for data in dataloader:
        i += 1
        inp, out = data
        inp = inp.to(device)
        out = out.to(device)

        out1, latent1 = model1(add_positional_encoding(inp))
        out2, latent2 = model2(add_positional_encoding(out))

        predicted_latent_2 = latent_translator(latent1)

        loss1 = F.mse_loss(out1, inp)
        weights1 = generate_color_weights_torch(out, colors).unsqueeze(1)
        weights2 = generate_color_weights_torch(out2, colors).unsqueeze(1)
        loss2 = stress_pattern_loss(out2, out, weights1*weights2)
        cosine_similarity = F.cosine_similarity(latent1, latent2)
        latent_loss = F.mse_loss(predicted_latent_2, latent2)
        if epoch < 0:
            loss = loss1 + loss2
        else:
            loss = loss1 + 100*loss2 + latent_loss*0.1 # -0.01*cosine_similarity.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if random.random() < 0.1:
            save_tensor_image(inp[0], f"{save_path}/{i}_inp.png")
            save_tensor_image(out[0], f"{save_path}/{i}_out.png")

            # output of encoder model1 which is trained on inp images
            save_tensor_image(out1[0], f"{save_path}/{i}_out1.png")
            
            # output of encoder model2 which is trained on out images
            save_tensor_image(out2[0], f"{save_path}/{i}_out2.png")
            decoded_predicted_latents = model2.from_latent(predicted_latent_2)

            # predicted output of translated latent from model1 to model2
            save_tensor_image(decoded_predicted_latents[0], f"{save_path}/{i}_decoded_predicted_latents.png")
            
            print(f"Epoch {epoch}, Loss1: {loss1.item()}, Loss2: {loss2.item()}")
            print(f"Latent loss: {latent_loss.item()}, Cosine similarity: {cosine_similarity.mean().item()}")
    model1.save(f"{save_path}/model1.pth")
    model2.save(f"{save_path}/model2.pth")
    latent_translator.save(f"{save_path}/latent_translator.pth")