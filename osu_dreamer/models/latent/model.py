
from dataclasses import dataclass
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.data.beatmap.encode import X_DIM, NUM_LABELS, HitSignals, CursorSignals
from osu_dreamer.data.load_audio import A_DIM

from osu_dreamer.common.rms_norm import RMSNorm

from .spec_features import SpecFeatures
from .unet import LayerArgs, UNetEncoder, UNetDecoder, layer

@dataclass
class LatentModelArgs:
    h_dim: int
    ae_args: LayerArgs
    
    style_head_dim: int
    style_heads: int

class AttnPool(nn.Module):
    def __init__(self, dim: int, out_dim: int, head_dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        h_dim = head_dim * n_heads
        self.scores = nn.Conv1d(dim, n_heads, 1)
        self.values = nn.Conv1d(dim, h_dim, 1)
        self.proj_out = nn.Linear(h_dim, out_dim)

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B O"]:
        a = self.scores(x).softmax(dim=-1)                  # B H L
        v = self.values(x).unflatten(1, (self.n_heads, -1)) # B H D L
        return self.proj_out(th.einsum('bhl,bhdl->bhd', a, v).flatten(1))

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
            AttnPool(args.h_dim, style_dim, args.style_head_dim, args.style_heads),
        )

        self.temporal_layer = layer(args.h_dim, style_dim, args.ae_args)
        self.temporal_head = nn.Sequential(
            nn.Conv1d(args.h_dim, emb_dim, 1),
            RMSNorm(emb_dim, affine=False),
        )

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
        z = self.temporal_head(self.temporal_layer(h, s))
        return z, s
    
    def decode_logits(
        self, 
        z: Float[Tensor, "B E l"],
        s: Float[Tensor, "B S"],
        *,
        audio: None | Float[Tensor, str(f"#B {A_DIM} L")] = None,
        skips: None | list[ Float[Tensor, "#B X _l"] ] = None,
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
        audio: None | Float[Tensor, str(f"#B {A_DIM} L")] = None,
        skips: None | list[ Float[Tensor, "#B X _l"] ] = None,
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