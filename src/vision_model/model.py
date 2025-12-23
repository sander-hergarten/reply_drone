"""U-Net model architecture for diffusion model."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (batch_size,) tensor of timesteps
        Returns:
            (batch_size, dim) tensor of embeddings
        """
        device = timesteps.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """Residual block with time embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class UNet(nn.Module):
    """U-Net architecture for diffusion model."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        time_emb_dim: int = 128,
        channel_multipliers: tuple = (1, 2, 4, 8),
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Encoder
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        channels = [in_channels] + [base_channels * m for m in channel_multipliers]

        # Encoder blocks
        for i in range(len(channels) - 1):
            self.encoder.append(
                ResidualBlock(channels[i], channels[i + 1], time_emb_dim)
            )

        # Bottleneck
        self.bottleneck = ResidualBlock(channels[-1], channels[-1], time_emb_dim)

        # Decoder blocks
        for i in reversed(range(len(channels) - 1)):
            self.decoder.append(
                ResidualBlock(channels[i + 1] + channels[i], channels[i], time_emb_dim)
            )

        # Output layer
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) noisy image
            timestep: (B,) timestep tensor
        Returns:
            (B, C, H, W) predicted clean image
        """
        # Time embedding
        t = self.time_mlp(timestep)

        # Encoder
        skip_connections = []
        for block in self.encoder:
            x = block(x, t)
            skip_connections.append(x)
            x = F.avg_pool2d(x, 2)

        # Bottleneck
        x = self.bottleneck(x, t)

        # Decoder
        for block in self.decoder:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x, t)

        # Output
        x = self.out_norm(x)
        x = F.siLU(x)
        x = self.out_conv(x)

        return x

