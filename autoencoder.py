import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys

import torch.nn as nn
import torch.optim as optim
from PIL import Image
import random
from autoencoder_model import ConvAutoencoder

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print_shape = True

def save_tensor_image(tensor, filename):
    # Convert tensor to image
    image = transforms.ToPILImage()(tensor)
    image.save(filename)




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
# initial data shape is (32, 3, 128, 128)

model = ConvAutoencoder()
print("Device: ", device)
model = model.to(torch.device(device))

num_epochs = 50
start_learning_rate = 8e-4
end_learning_rate = 8e-6
# Create linear learning rate scheduler from start to end over num_epochs
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.AdamW(model.parameters(), lr=start_learning_rate)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, .01, num_epochs)

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
    scheduler.step()  # Update learning rate
    torch.save(model.state_dict(), f"outputs/autoencoder{epoch}.pth")

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}') 

