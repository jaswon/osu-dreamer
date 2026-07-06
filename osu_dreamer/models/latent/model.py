
from dataclasses import dataclass
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.data.beatmap.encode import X_DIM, NUM_LABELS, HitSignals, CursorSignals
from osu_dreamer.data.load_audio import A_DIM

from osu_dreamer.modules.spec_features import SpecFeatures

from .unet import AEArgs, UNetEncoder, UNetDecoder
from .label_predictor import LabelPredictor, LabelPredictorArgs

@dataclass
class LatentModelArgs:
    h_dim: int
    ae_args: AEArgs
    label_args: LabelPredictorArgs

class LatentModel(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        n_downs: int,
        stride: int,
        args: LatentModelArgs,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_downs = n_downs
        self.stride = stride
        self.chunk_size = stride ** n_downs

        self.chart_encoder = nn.Sequential( nn.Conv1d(X_DIM, args.h_dim, 1), UNetEncoder(args.h_dim, n_downs, stride, args.ae_args) )
        self.audio_encoder = nn.Sequential( SpecFeatures(A_DIM, args.h_dim), UNetEncoder(args.h_dim, n_downs, stride, args.ae_args) )
        self.mu = nn.Conv1d(args.h_dim, emb_dim, 1)
        self.logvar = nn.Conv1d(args.h_dim, emb_dim, 1)

        self.proj_emb = nn.Conv1d(emb_dim, args.h_dim, 1)
        self.decoder = UNetDecoder(args.h_dim, n_downs, stride, args.ae_args)
        self.proj_out = nn.Conv1d(args.h_dim, X_DIM, 1)

        self.label_predictor = LabelPredictor(emb_dim, NUM_LABELS, args.label_args)

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        z: Float[Tensor, "B E l"],
    ) -> tuple[
        Float[Tensor, str(f"B {X_DIM} L")], 
        Float[Tensor, str(f"B {NUM_LABELS}")],
    ]:
        return (
            self.decode_logits(audio, z),
            self.label_predictor(z),
        )
    
    def param_encode(
        self,
        true_chart: Float[Tensor, str(f"B {X_DIM} L")],
    ) -> tuple[
        Float[Tensor, "B E l"],
        Float[Tensor, "B E l"],
    ]:
        _, h = self.chart_encoder(true_chart)
        return self.mu(h), self.logvar(h)
    
    def decode_logits(
        self, 
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        z: Float[Tensor, "B E l"],
    ) -> Float[Tensor, str(f"B {X_DIM} L")]:
        skips, _ = self.audio_encoder(audio)
        return self.proj_out(self.decoder(skips, self.proj_emb(z))[:,:,:audio.size(-1)])
    
    @th.no_grad
    def encode_chart(self, chart: Float[Tensor, str(f"B {X_DIM} L")]) -> Float[Tensor, "B D l"]:
        _, h = self.chart_encoder(chart)
        return self.mu(h)
    
    @th.no_grad
    def encode_audio(self, audio: Float[Tensor, str(f"B {A_DIM} L")]) -> Float[Tensor, "B A l"]:
        _, h = self.audio_encoder(audio)
        return h
    
    @th.no_grad
    def decode(
        self,
        a: Float[Tensor, "*B A L"],
        z: Float[Tensor, "B D l"],
    ) -> tuple[
        Float[Tensor, str(f"B {X_DIM} L")], 
        Float[Tensor, str(f"B {NUM_LABELS}")],
    ]:
        pred_logits = self.decode_logits(a.expand(z.size(0),-1,-1), z)
        pred_chart = th.cat([
            pred_logits[:, HitSignals].sigmoid(),
            pred_logits[:, CursorSignals],
        ], dim=1)
        pred_labels = self.label_predictor(z).clamp(0, 10)
        return pred_chart, pred_labels