"""
file takes 1 argmument, the path to the dataset
use /low_res_inputs as the path to the dataset
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys

import torch.nn as nn
import torch.optim as optim
from PIL import Image
import random
from autoencoder_model import ConvAutoencoder


# create a library of training parameters
l_lr_n = 5
l_lr_s = [8e-3, 8e-4, 8e-5, 8e-6, 8e-7]
l_lr_e = [8e-5, 8e-6, 8e-7, 8e-8, 8e-9]
l_epochs_n = 5
l_epochs = [50, 100, 150, 200, 250]
param_lib = {}
start_loop = 3
for e in l_epochs:
    for lr in range(len(l_lr_s)):
        param_lib[start_loop] = (e, l_lr_s[lr], l_lr_e[lr])
        start_loop += 1

# go through the library and print the key and the values
for key, value in param_lib.items():
    print(key, value)

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

'''
model = ConvAutoencoder()
print("Device: ", device)
model = model.to(torch.device(device))

num_epochs = 500
start_learning_rate = 8e-3
end_learning_rate = 8e-5

# Create linear learning rate scheduler from start to end over num_epochs
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.AdamW(model.parameters(), lr=start_learning_rate)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_learning_rate, end_learning_rate, num_epochs)

i = 0
'''

# Training loop
for key, params in param_lib.items():
    num_epochs, start_learning_rate, end_learning_rate = params
    # reset the model
    model = ConvAutoencoder()
    model = model.to(torch.device(device))
    criterion = nn.MSELoss()  # Mean Squared Error loss
    # reset the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=start_learning_rate)
    # reset the scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_learning_rate, end_learning_rate, num_epochs)
    i = 0
    print(f"Training cycle {key}, epochs: {num_epochs}, start_lr: {start_learning_rate}, end_lr: {end_learning_rate}")

    for epoch in range(num_epochs):
        for data in dataloader:
            img, _ = data  # Assuming your DataLoader returns (image, label) pairs
            img = img.mean(dim=1, keepdim=True)  # Convert to grayscale (1 channel)
            img = img.to(device)  # Move image to the device (GPU if available)

            # Forward pass
            output = model(img)
            loss = criterion(output, img)  # Compare reconstructed image with original

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i+=1
            if i % 10000 == 0:
                print("Loss: ", loss.item())
                save_tensor_image(output[0], f'encoder_outputs/{key}_{i}_output.png')
                save_tensor_image(img[0], f'encoder_outputs/{key}_{i}_input.png')
        scheduler.step()  # Update learning rate
        torch.save(model.state_dict(), f"encoder_outputs/{key}_autoencoder{epoch}.pth")

        print(f'training round {key} Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}') 
    
    # create a txt file if it does not exist and save the loss
    with open(f'encoder_outputs/all_losses.txt', 'a') as f:
        f.write(f'Training cycle {key}: loss {loss.item()}\n')

