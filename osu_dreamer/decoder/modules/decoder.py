
from jaxtyping import Float

from dataclasses import dataclass

import torch as th
from torch import nn, Tensor

@dataclass
class DecoderArgs:
    ...

class Decoder(nn.Module):
    def __init__(
        self,
        args: DecoderArgs,
    ):
        super().__init__()

    def forward(
        self,
        embeds: Float[Tensor, "B N E"],
        embed_positions: Float[Tensor, "B N"],
        ctx: Float[Tensor, "B L H"],
        ctx_positions: Float[Tensor, "B L"],
        conditioning: Float[Tensor, "B C"],
    ) -> Float[Tensor, "B N E"]:
        return x