# models/autoencoder.py

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Простіший convolutional autoencoder для зображень 3x128x128.

    Encoder:
      128 -> 64 -> 32 -> 16 -> 8 (простір ознак 256 x 8 x 8)
    Decoder:
      8 -> 16 -> 32 -> 64 -> 128

    forward(x) -> (x_recon, z), де:
      z - латентний вектор розмірності latent_dim
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # --- Encoder ---
        self.encoder_conv = nn.Sequential(
            # 3 x 128 x 128 -> 32 x 64 x 64
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 32 x 64 x 64 -> 64 x 32 x 32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 64 x 32 x 32 -> 128 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 128 x 16 x 16 -> 256 x 8 x 8
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.feature_shape = (256, 8, 8)
        self.flatten = nn.Flatten()

        # 256 * 8 * 8 = 16384
        self.fc_enc = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 8 * 8)

        # --- Decoder ---
        self.decoder_conv = nn.Sequential(
            # 256 x 8 x 8 -> 128 x 16 x 16
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(inplace=True),

            # 128 x 16 x 16 -> 64 x 32 x 32
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(inplace=True),

            # 64 x 32 x 32 -> 32 x 64 x 64
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(inplace=True),

            # 32 x 64 x 64 -> 3 x 128 x 128
            nn.ConvTranspose2d(
                32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),  # вихід в [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)
        h_flat = self.flatten(h)
        z = self.fc_enc(h_flat)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h_flat = self.fc_dec(z)
        h = h_flat.view(-1, *self.feature_shape)
        x_recon = self.decoder_conv(h)
        return x_recon

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
