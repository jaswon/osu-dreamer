
from dataclasses import dataclass
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.data.beatmap.encode import X_DIM, NUM_LABELS, HitSignals, CursorSignals
from osu_dreamer.data.load_audio import A_DIM

from osu_dreamer.modules.spec_features import SpecFeatures
from osu_dreamer.modules.ae import AEArgs, Encoder, Decoder

from .label_predictor import LabelPredictor, LabelPredictorArgs

@dataclass
class LatentModelArgs:
    a_dim: int
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

        self.encoder = Encoder(            X_DIM,      -1, n_downs, stride, args.ae_args)
        self.decoder = Decoder(args.a_dim, X_DIM, emb_dim, n_downs, stride, args.ae_args)
        self.latent_spec_features = SpecFeatures(A_DIM, args.a_dim)
        self.mu     = nn.Conv1d(args.ae_args.h_dim, emb_dim, 1)
        self.logvar = nn.Conv1d(args.ae_args.h_dim, emb_dim, 1)
        self.label_predictor = LabelPredictor(emb_dim, NUM_LABELS, args.label_args)

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        true_chart: Float[Tensor, str(f"B {X_DIM} L")],
    ) -> tuple[
        Float[Tensor, str(f"B {X_DIM} L")], 
        Float[Tensor, str(f"B {NUM_LABELS}")],
        Float[Tensor, ""],
    ]:
        h = self.encoder(true_chart)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + th.exp(0.5 * logvar) * th.randn_like(mu)
        kl = (0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)).sum(dim=1).mean()
        return (
            self.decoder(self.latent_spec_features(audio), z),
            self.label_predictor(z),
            kl,
        )
    
    @th.no_grad
    def encode(
        self,
        chart: Float[Tensor, str(f"B {X_DIM} L")],
    ) -> Float[Tensor, "B D l"]:
        return self.mu(self.encoder(chart))
    
    @th.no_grad
    def decode(
        self,
        a: Float[Tensor, "*B A L"],
        z: Float[Tensor, "B D l"],
    ) -> tuple[
        Float[Tensor, str(f"B {X_DIM} L")], 
        Float[Tensor, str(f"B {NUM_LABELS}")],
    ]:
        logits = self.decoder(self.latent_spec_features(a.expand(z.size(0),-1,-1)), z)
        pred_labels = self.label_predictor(z).clamp(0, 10)
        pred_chart = th.cat([
            logits[:, HitSignals].sigmoid(),
            logits[:, CursorSignals],
        ], dim=1)
        return pred_chart, pred_labels