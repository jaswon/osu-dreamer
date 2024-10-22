
from dataclasses import dataclass

from jaxtyping import Float

from torch import nn, Tensor

from osu_dreamer.data.load_audio import A_DIM

from osu_dreamer.modules.wavenet import WaveNet, WaveNetArgs

@dataclass
class AudioFeatureArgs(WaveNetArgs):
    h_dim: int

class AudioFeatures(nn.Module):
    def __init__(
        self,
        dim: int,
        args: AudioFeatureArgs,
    ):
        super().__init__()

        class block(nn.Module):
            def __init__(self, depth: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv1d(dim, args.h_dim, 1),
                    nn.SiLU(),
                    nn.Conv1d(args.h_dim, dim, 1),
                )
        
            def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L"]:
                return x + self.net(x)

        self.proj_in = nn.Conv1d(A_DIM, dim, 1)
        self.net = WaveNet(dim, None, args, block)

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
    ) -> Float[Tensor, "B D L"]:
        return self.net(self.proj_in(audio), None)