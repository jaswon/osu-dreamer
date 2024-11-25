
from jaxtyping import Float

from torch import nn, Tensor

class ResNet(nn.Module):
    def __init__(
        self,
        dim: int,
        layers: list[nn.Module],
    ):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList([ nn.GroupNorm(1, dim) for _ in layers ])
        self.post_norm = nn.GroupNorm(1, dim)

    def forward(
        self, 
        x: Float[Tensor, "B X L"], 
        *args, **kwargs,
    ) -> Float[Tensor, "B X L"]:
        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x),*args,**kwargs)
        return self.post_norm(x)
