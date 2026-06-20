
from torch import nn

class Res(nn.Sequential):
    def forward(self, x):
        return x + super().forward(x)