
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from einops import rearrange

from osu_dreamer.common.residual import ResStack

from osu_dreamer.data.beatmap.encode import CursorSignals

class WaveNet(ResStack):
    def __init__(self, dim: int, num_stacks: int, stack_depth: int):
        super().__init__(dim, [
            nn.Sequential(
                nn.ZeroPad1d((2**d,0)),
                nn.Conv1d(dim, 2*dim, 2, dilation=2**d),
                nn.GLU(dim=1),
            )
            for _ in range(num_stacks)
            for d in range(stack_depth)
        ])

CRITIC_FEATURES = 8
def critic_features(x: Float[Tensor, "B X L"]) -> Float[Tensor, str(f"B {CRITIC_FEATURES} L")]:
    cursor = x[:,CursorSignals]
    cursor_diff = F.pad(cursor[...,1:] - cursor[...,:-1], (1,0), mode='replicate')
    return th.cat([ x, cursor_diff ], dim=1)

@dataclass
class CriticArgs:
    audio_h_dim: int
    audio_num_stacks: int
    audio_stack_depth: int

    h_dim: int
    num_stacks: int
    stack_depth: int

class Critic(nn.Module):
    def __init__(
        self,
        a_dim: int,
        x_dim: int,
        args: CriticArgs,
    ):
        super().__init__()

        self.audio_pre = nn.Sequential(
            nn.Conv1d(a_dim, a_dim, 5,1,2, groups=a_dim),
            nn.Conv1d(a_dim, 2*args.audio_h_dim, 1),
            nn.GLU(dim=1),
            WaveNet(args.audio_h_dim, args.audio_num_stacks, args.audio_stack_depth)
        )

        self.net = nn.Sequential(
            nn.Conv1d(args.audio_h_dim + CRITIC_FEATURES, args.h_dim, 1),
            WaveNet(args.h_dim, args.num_stacks, args.stack_depth),
            nn.Conv1d(args.h_dim, x_dim, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.modules.conv._ConvNd):
                th.nn.utils.spectral_norm(m)

        # receptive field
        self.rf = 1 + (2**args.stack_depth-1)*args.num_stacks

    def grad_norm(self, a: Float[Tensor, "B A L"], x: Float[Tensor, "B X L"]) -> Float[Tensor, ""]:
        L = (x.size(-1) // self.rf) * self.rf
        x = x[:,:,:L].requires_grad_()
        a = a[:,:,:L]
        grad = th.autograd.grad(self(a, x)[:,:,::self.rf].sum(), x, create_graph=True)[0]
        grad_norm = rearrange(grad.pow(2), 'b x (f l) -> b (x f) l', f = self.rf).sum(1)
        return grad_norm.mean()

    def forward(
        self, 
        audio: Float[Tensor, "B A L"],
        cursor: Float[Tensor, "B X L"],
    ) -> Float[Tensor, "B X L"]:
        return self.net(th.cat([self.audio_pre(audio), critic_features(cursor)], dim=1))