import torch.nn as nn
import torch

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.print_shape = True

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
            nn.Conv2d(32, 32, 3, stride=3, padding=1),  # 5th conv layer
            nn.ReLU()
        )


        # Decoder 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=3, padding=1, output_padding=1),  # 1st deconv layer
            nn.ReLU(),
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

        encoded = encoded + torch.randn_like(encoded) * 0.01

        if self.print_shape:
            print("Encoded shape: ", encoded.shape)
            self.print_shape = False
            
       # print("Encoded shape: ", encoded.shape)
        decoded = self.decoder(encoded)
        return decoded
