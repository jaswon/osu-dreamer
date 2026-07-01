
from torch import nn

class Res(nn.Sequential):
    def __init__(self, *args, alpha: int = 1):
        super().__init__(*args)
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * x + super().forward(x)