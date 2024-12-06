
from collections.abc import Sequence
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.mingru import complex_log

    
class RandomFourierFeatures(nn.Module):
    def __init__(self, dim: int, n_feats: int):
        super().__init__()
        self.register_buffer('f', 2 * th.pi * th.randn(dim, n_feats))
        self.register_buffer('p', 2 * th.pi * th.rand(n_feats))

    def forward(self, x: Float[Tensor, "B C"]) -> Float[Tensor, "B N"]:
        return 2**.5 * th.cos(x @ self.f + self.p)
    
def min_gru(
    h: Float[Tensor, "... L"], 
    g: Float[Tensor, "... L"],
) -> Float[Tensor, "... L"]:
    log_scale = .5 * th.log(1/th.cosh(g)+1) # mp
    log_coeffs = -F.softplus(g) + log_scale
    log_values = -F.softplus(-g) + log_scale + complex_log(h)

    # heinsen associative scan (log-space)
    a_star = log_coeffs.cumsum(dim=-1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=-1)
    log_h = a_star + log_h0_plus_b_star
    
    return log_h.exp().real

class minGRU2(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim%2==0
        self.fb_hg = MP.Conv1d(dim, dim*2, 1)

    def forward(
        self,
        x: Float[Tensor, "B H L"],
    ) -> Float[Tensor, "B H L"]:
        fore_hg, back_hg = self.fb_hg(x).chunk(2, dim=1)
        return MP.cat([
            min_gru(*fore_hg.chunk(2, dim=1)),
            min_gru(*back_hg.flip(2).chunk(2, dim=1)).flip(2),
        ], dim=1)
    
class Seq(nn.Module):
    def __init__(self, dim: int, h_dim: int):
        super().__init__()
        self.h = nn.Sequential(
            MP.SiLU(),
            MP.Conv1d(dim, h_dim, 1),
            MP.Conv1d(h_dim, h_dim, 3,1,1, groups=h_dim),
            MP.SiLU(),
            minGRU2(h_dim),
            MP.PixelNorm(),
        )
        self.g = nn.Sequential(
            MP.SiLU(),
            MP.Conv1d(dim, h_dim, 1),
            MP.SiLU(),
        )
        self.out = MP.Conv1d(h_dim, dim, 1)

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L"]:
        return self.out(self.h(x) * self.g(x))
    
class ResNet(nn.Module):
    def __init__(self, nets: Sequence[nn.Module]):
        super().__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, x: Float[Tensor, "B D L"], *args, **kwargs) -> Float[Tensor, "B D L"]:
        for net in self.nets:
            x = MP.pixel_norm(x)
            x = MP.add(x, net(x,*args,**kwargs), t=.1)
        return x