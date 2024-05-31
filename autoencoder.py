import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys

import torch.nn as nn
import torch.optim as optim
from PIL import Image
import random

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

def save_tensor_image(tensor, filename):
    # Convert tensor to image
    image = transforms.ToPILImage()(tensor)
    image.save(filename)


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),  # 1st conv layer, input channels = 3 (for RGB images)
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), # 2nd conv layer
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 3rd conv layer
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # 4th conv layer
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
        )

        self.linear_output = nn.Linear(256, 2048)

        # Decoder 
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output layer, sigmoid for pixel values between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        #print("Encoded shape: ", encoded.shape)
        encoded = self.linear_output(encoded)
        encoded = encoded.view(-1, 32, 8, 8)
        decoded = self.decoder(encoded)
        return decoded


# 1. Define Transformations (same as before)
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Create Dataset (modified)
# We use the same ImageFolder class but ignore the labels it provides.
image_dataset = datasets.ImageFolder(root=sys.argv[1], transform=data_transforms)

# 3. Create DataLoader (same as before)
dataloader = DataLoader(image_dataset, batch_size=32, shuffle=True, num_workers=4)

model = ConvAutoencoder()
print("Device: ", device)
model = model.to(torch.device(device))

num_epochs = 10
learning_rate = 8e-4
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

i = 0

# Training loop
for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data  # Assuming your DataLoader returns (image, label) pairs
        img = img.to(device)  # Move image to the device (GPU if available)

        # Forward pass
        output = model(img)
        loss = criterion(output, img)  # Compare reconstructed image with original

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i+=1
        if i % 1000 == 0:
            print("Loss: ", loss.item())
            save_tensor_image(output[0], f'outputs/{i}_output.png')
            save_tensor_image(img[0], f'outputs/{i}_input.png')

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}') 
