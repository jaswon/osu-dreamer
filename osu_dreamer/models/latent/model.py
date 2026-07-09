
from dataclasses import dataclass
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.data.beatmap.encode import X_DIM, NUM_LABELS, HitSignals, CursorSignals
from osu_dreamer.data.load_audio import A_DIM

from osu_dreamer.modules.spec_features import SpecFeatures

from .unet import LayerArgs, UNetEncoder, UNetDecoder, layer

@dataclass
class LatentModelArgs:
    h_dim: int
    ae_args: LayerArgs

class LatentModel(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        style_dim: int,
        n_downs: int,
        stride: int,
        args: LatentModelArgs,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.style_dim = style_dim
        self.a_dim = args.h_dim
        self.chunk_size = stride ** n_downs

        self.chart_encoder = nn.Sequential( nn.Conv1d(X_DIM, args.h_dim, 1), UNetEncoder(args.h_dim, n_downs, stride, args.ae_args) )
        self.audio_encoder = nn.Sequential( SpecFeatures(A_DIM, args.h_dim), UNetEncoder(args.h_dim, n_downs, stride, args.ae_args) )
        self.style_head = nn.Sequential(
            layer(args.h_dim, 0, args.ae_args),
            nn.Conv1d(args.h_dim, style_dim, 1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
        )

        self.param_layer = layer(args.h_dim, style_dim, args.ae_args)
        self.param_proj = nn.Conv1d(args.h_dim, emb_dim, 1)

        self.proj_emb = nn.Conv1d(emb_dim, args.h_dim, 1)
        self.decoder = UNetDecoder(args.h_dim, style_dim, n_downs, stride, args.ae_args)
        self.proj_out = nn.Conv1d(args.h_dim, X_DIM, 1)

        self.label_predictor = nn.Sequential(
            nn.Linear(style_dim, args.h_dim),
            nn.SiLU(),
            nn.Linear(args.h_dim, NUM_LABELS),
        )

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        z: Float[Tensor, "B E l"],
        s: Float[Tensor, "B S"],
    ) -> tuple[
        Float[Tensor, str(f"B {X_DIM} L")], 
        Float[Tensor, str(f"B {NUM_LABELS}")],
    ]:
        return (
            self.decode_logits(z, s, audio=audio),
            self.label_predictor(s),
        )
    
    def encode_chart(
        self,
        chart: Float[Tensor, str(f"B {X_DIM} L")],
    ) -> tuple[
        Float[Tensor, "B E l"],
        Float[Tensor, "B S"],
    ]:
        _, h = self.chart_encoder(chart)
        s = self.style_head(h)

        # `s.detach()`: the style branch must earn its content via its own losses, 
        # not serve as a reconstruction side-channel
        z = self.param_proj(self.param_layer(h, s.detach()))

        # DC projection: remove per-channel window means so time-invariant
        # information cannot live in `z` and must route through `s`
        z = z - z.mean(dim=-1, keepdim=True)
        return z, s
    
    def decode_logits(
        self, 
        z: Float[Tensor, "B E l"],
        s: Float[Tensor, "B S"],
        *,
        audio: None | Float[Tensor, str(f"B {A_DIM} L")] = None,
        skips: None | list[ Float[Tensor, "B X _l"] ] = None,
    ) -> Float[Tensor, str(f"B {X_DIM} L")]:
        if skips is None:
            skips, _ = self.audio_encoder(audio)
            assert skips is not None
        return self.proj_out(self.decoder(skips, self.proj_emb(z), s))
    
    @th.no_grad
    def decode(
        self,
        z: Float[Tensor, "B D l"],
        s: Float[Tensor, "B S"],
        *,
        audio: None | Float[Tensor, str(f"B {A_DIM} L")] = None,
        skips: None | list[ Float[Tensor, "B X _l"] ] = None,
    ) -> tuple[
        Float[Tensor, str(f"B {X_DIM} L")], 
        Float[Tensor, str(f"B {NUM_LABELS}")],
    ]:
        pred_logits = self.decode_logits(z, s, audio=audio, skips=skips)
        pred_chart = th.cat([
            pred_logits[:, HitSignals].sigmoid(),
            pred_logits[:, CursorSignals],
        ], dim=1)
        pred_labels = self.label_predictor(s).clamp(0, 10)
        return pred_chart, pred_labels