
from typing import Any
from jaxtyping import Float, Int

import torch as th
from torch import Tensor, nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.utils.tensorboard.writer import SummaryWriter

from osu_dreamer.data.labels import NUM_LABELS
from osu_dreamer.data.load_audio import A_DIM

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.muon import Muon


from .data.module import Batch
from .data.tokens import Token, TokenType, encode, BOS, VOCAB_SIZE, DIFF, PAD

from .modules.label import LabelEmbedding, LabelEmbeddingArgs
from .modules.decoder import Decoder, DecoderArgs
from .modules.encoder import Encoder, EncoderArgs

from .make_batch import make_batch


def focal_loss(
    inputs: Float[Tensor, "B D ..."],
    target: Int[Tensor, "B ..."],
    gamma: float,
    weight: None | Float[Tensor, "D"] = None,
    *args, **kwargs,
) -> Float[Tensor, "B ..."]:
    logpt = F.log_softmax(inputs, dim=1)
    inputs = (1 - logpt.exp()).pow(gamma) * logpt
    return F.nll_loss(inputs, target, weight, reduction='none', *args, **kwargs)

    
class Model(pl.LightningModule):
    def __init__(
        self,

        # training parameters
        seq_len: int,
        opt_args: dict[str, Any],
        focal_gamma: float,
        max_token_numel: int,

        # model hparams
        ctx_dim: int,
        encoder_dim: int,
        encoder_args: EncoderArgs,

        label_dim: int,
        label_emb_args: LabelEmbeddingArgs,

        embed_dim: int,
        decoder_args: DecoderArgs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # training params
        self.seq_len = seq_len
        self.opt_args = opt_args
        self.focal_gamma = focal_gamma
        self.max_token_numel = max_token_numel

        # model
        self.embed_dim = embed_dim
        self.audio_encoder = nn.Sequential(
            MP.Linear(A_DIM, encoder_dim),
            Encoder(encoder_dim, encoder_args),
            MP.Linear(encoder_dim, ctx_dim),
        )

        self.type_head = nn.Sequential(
            MP.Linear(embed_dim, 2), # { token, timestamp }
            MP.Gain(),
        )

        self.timestamp_emb = MP.Embedding(seq_len, embed_dim)
        self.timestamp_head = nn.Sequential(
            MP.Linear(embed_dim, seq_len),
            MP.Gain(),
        )
        self.timestamp_head[0].weight = self.timestamp_emb.weight

        self.token_emb = MP.Embedding(VOCAB_SIZE, embed_dim)
        self.token_head = nn.Sequential(
            MP.Linear(embed_dim, VOCAB_SIZE),
            MP.Gain(),
        )
        self.token_head[0].weight = self.token_emb.weight

        self.label_emb = LabelEmbedding(label_dim, label_emb_args)
        self.label_head = nn.Linear(embed_dim, NUM_LABELS)

        self.decoder = Decoder(
            embed_dim,
            ctx_dim,
            label_dim,
            decoder_args,
            self.seq_len,
        )

    def forward(
        self,
        labels: Float[Tensor, str(f"1 {NUM_LABELS}")],
        audio: Float[Tensor, str(f"1 {A_DIM} L")],
        types: Int[Tensor, "1 N"],
        tokens: Int[Tensor, "1 N"],
        timestamps: Float[Tensor, "1 N"],
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        
        D = audio.device
        ctx = self.audio_encoder(audio.transpose(1,2))[0] # L H
        b_ctx_idxs, b_types, b_token_idxs, b_tokens = make_batch(
            types[0], tokens[0], timestamps[0],
            seq_len = self.seq_len,
            num_frames = ctx.size(0),
            max_token_numel = self.max_token_numel,
        )
        batch_size = b_ctx_idxs.size(0)
        b_ctx = ctx[b_ctx_idxs]

        # randomly mask labels for training
        b_labels = labels.repeat(batch_size, 1)
        label_embs = self.label_emb(th.where(th.rand_like(b_labels) < .5, -1, b_labels))

        # prepare decoder prelude
        b_prelude_embs = self.token_emb(th.tensor([DIFF, BOS], device=D)).repeat(batch_size, 1, 1)
        b_prelude_idxs = b_ctx_idxs[:,:1].repeat(1, 2)

        # prepare decoder inputs
        b_token_embs = th.gather(
            th.stack([
                self.token_emb(b_tokens),
                self.timestamp_emb(th.clamp(b_token_idxs - b_ctx_idxs[:,:1], min=0)),
            ], dim=0),
            dim = 0,
            index = b_types[None,:,:,None].expand(-1,-1,-1,self.embed_dim),
        )[0]

        h = self.decoder(
            x = th.cat([b_prelude_embs, b_token_embs], dim=1),
            x_t = th.cat([b_prelude_idxs, b_token_idxs], dim=1),
            ctx = b_ctx,
            ctx_t = b_ctx_idxs,
            c = label_embs,
        ) # B N+2 E

        pred_labels = self.label_head(h[:,0]) # B NUM_LABELS
        label_loss = (b_labels - pred_labels).pow(2).mean()

        pred_embs = h[:,1:-1] # B N E
        pred_typs = self.type_head(pred_embs) # B N 2
        type_loss = focal_loss(
            pred_typs.flatten(0,1),
            b_types.flatten(),
            gamma = self.focal_gamma,
        ).mean()

        pred_token_logits = self.token_head(pred_embs) # B N V
        token_loss = focal_loss(
            pred_token_logits[b_types == 0],
            b_tokens[b_types == 0],
            gamma = self.focal_gamma,
            ignore_index = PAD,
        ).mean()

        pred_timestamp_logits = self.timestamp_head(pred_embs) # B N S
        pred_timestamp_cdf = pred_timestamp_logits.softmax(dim=-1).cumsum(dim=-1) # B N S
        true_timestamp_cdf = (b_token_idxs[:,:,None] >= b_ctx_idxs[:,None,:]).long() # B N S
        timestamp_loss = (
            pred_timestamp_cdf[b_types == 1]
            - true_timestamp_cdf[b_types == 1]
        ).pow(2).mean()

        loss = label_loss + type_loss + token_loss + timestamp_loss
        return loss, {
            "loss": loss.detach(),
            "label": label_loss.detach(),
            "type": type_loss.detach(),
            "token": token_loss.detach(),
            "timestamp": timestamp_loss.detach(),
            "b_tokens.numel": th.tensor(b_tokens.numel(), dtype=th.float),
            "audio_len": th.tensor(audio.size(-1), dtype=th.float),
        }
    
    @th.no_grad
    def sample(
        self,
        audio: Float[Tensor, str(f"{A_DIM} L")],
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
        time_budget: int | float = float('inf'), # max allowed time (sec)
        show_progress: bool = False,
    ) -> tuple[
        list[list[Token|float]],                # list of B lists of tokens and timestamps
        Float[Tensor, str(f"B {NUM_LABELS}")],  # predicted labels
    ]:
        from .sample import sample
        return sample(self, audio, labels, time_budget, show_progress)


#
#
# =============================================================================
# MODEL TRAINING
# =============================================================================
#
#

    def configure_optimizers(self):
        return Muon(self.parameters(), **self.opt_args)

    def training_step(self, batch: Batch, batch_idx):
        loss, log_dict = self(*batch)
        self.log_dict({ f"train/{k}": v for k,v in log_dict.items() })
        return loss
    
    def on_after_backward(self):
        self.log("train/grad_l2", sum(
            p.grad.detach().norm(2).item() ** 2
            for p in self.parameters()
            if p.grad is not None
        ) ** .5)
 
    def validation_step(self, batch: Batch, batch_idx, *args, **kwargs):
        _, log_dict = self(*batch)
        self.log_dict({ f"val/{k}": v for k,v in log_dict.items() })

        if batch_idx == 0:
            self.plot_val(batch)

    @th.no_grad()
    def plot_val(self, batch: Batch):

        true_label = batch[0][0]
        audio = batch[1][0]
        label = true_label.repeat(2,1)
        label = th.where(th.rand_like(label) < .5, -1, label)

        exp: SummaryWriter = self.logger.experiment # type: ignore

        def f_label(label: Float[Tensor, str(f"{NUM_LABELS}")]):
            sr, ar, od, cs, hp = [ round(l.item(), ndigits=1) for l in label ]
            return f'{sr=:>4} {ar=:>4} {od=:>4} {cs=:>4} {hp=:>4}'

        samples, pred_labels = self.sample(audio, label, time_budget=10)
        for i, (sample, pred_label) in enumerate(zip(samples, pred_labels)):
            sample_text = '\n'.join([
                f'true: {f_label(true_label)}',
                f'pred: {f_label(pred_label)}',
                '',
            ] + [ str(event) for event in sample ])
            exp.add_text(f'sample/{i}', sample_text, global_step=self.global_step)