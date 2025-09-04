
import torch.nn as nn

from einops.layers.torch import Rearrange, Reduce

class ResidualBlock(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1),
        )
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(x + self.conv(x))

class SimpleAudioEncoder(nn.Module):
    def __init__(self, h_dim: int, num_layers: int = 4):
        super().__init__()
        
        layers = [
            Rearrange('b d l -> b 1 d l'),
            nn.Conv2d(1, h_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        
        for i in range(num_layers):
            layers.append(ResidualBlock(h_dim))
            # Downsample by 2 in time and frequency every 2 layers
            if i % 2 == 1:
                layers.append(nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=2, padding=1))
            
        layers.append(Reduce('b d f l -> b d l', 'mean'))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)
