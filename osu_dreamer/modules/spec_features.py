
from jaxtyping import Float

from torch import nn, Tensor

from einops.layers.torch import Rearrange
from einops import rearrange

from osu_dreamer.data.load_audio import A_DIM
import osu_dreamer.modules.mp as MP

class sepConv(nn.Module):
    def __init__(self, shape: str, h_dim: int, *args):
        super().__init__()
        self.shape = shape
        self.conv = MP.Conv1d(h_dim, h_dim, *args, groups=h_dim)

    def forward(self, x: Float[Tensor, "B H F L"]) -> Float[Tensor, "B H _F L"]:
        b = x.size(0)
        x = rearrange(x, f'b h f l -> {self.shape}')
        x = self.conv(x)
        return rearrange(x, f'{self.shape} -> b h f l', b=b)
    
conf = [
    (3,2),
    (5,3),
    (5,3),
    (7,4),
]

class SpecFeatures(nn.Sequential):
    def __init__(self, h_dim: int, conf: list[tuple[int, int]] = conf):
        total_features = 1
        for i, (t,f) in enumerate(conf):
            total_features *= f
            assert t % 2 == 1, f"conf[{i},0] must be odd, got {t}"
        assert total_features == A_DIM, ""

        super().__init__(
            Rearrange('b f l -> b 1 f l'),
            MP.Conv2d(1, h_dim, 1),
            *(
                layer for t,f in conf
                for conv in [
                    sepConv('(b f) h l', h_dim, t,1,t//2), 
                    sepConv('(b l) h f', h_dim, f,f,1),
                ]
                for layer in [
                    conv, 
                    MP.Conv2d(h_dim, h_dim, 1),
                    MP.SiLU(),
                ]
            ),
            Rearrange('b h 1 l -> b h l'),
        )