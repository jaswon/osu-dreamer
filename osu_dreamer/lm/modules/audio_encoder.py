
import torch.nn as nn

from einops.layers.torch import Rearrange, Reduce

class SimpleAudioEncoder(nn.Module):
    def __init__(self, h_dim: int, num_layers: int = 2):
        super().__init__()
        
        layers = [
            Rearrange('b d l -> b 1 d l'),
            nn.Conv2d(1, h_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        
        for _ in range(num_layers):
            layers.extend([
                nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1),
                nn.SiLU(),
            ])
            
        layers.append(Reduce('b d f l -> b d l', 'mean'))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)
