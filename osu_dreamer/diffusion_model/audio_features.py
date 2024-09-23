
from jaxtyping import Float

from torch import nn, Tensor

from osu_dreamer.data.load_audio import A_DIM

from ..modules.wavenet import WaveNet, WaveNetArgs

class AudioFeatureArgs(WaveNetArgs):
    pass

class AudioFeatures(nn.Module):
    def __init__(
        self,
        dim: int,
        args: AudioFeatureArgs,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(A_DIM, dim, 1),
            WaveNet(dim, args, lambda dim: nn.Sequential(
                nn.Conv1d(dim, dim, 3,1,1, groups=dim),
                nn.Conv1d(dim, dim, 1),
                nn.SiLU(),
                nn.Conv1d(dim, dim, 1),
            )),
        )

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
    ) -> Float[Tensor, "B A L"]:
        return self.net(audio)