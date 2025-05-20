
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

from osu_dreamer.audio_encoder.model import Model as AudioEncoder

from .data.module import Batch
from .data.tokens import Token, TokenType, encode, BOS, VOCAB_SIZE, DIFF, T0

from .modules.label import LabelEmbedding, LabelEmbeddingArgs
from .modules.decoder import Decoder, DecoderArgs

from .make_batch import make_batch


def focal_loss(
    inputs: Float[Tensor, "B D ..."],
    target: Int[Tensor, "B ..."],
    gamma: float,
    weight: None | Float[Tensor, "D"] = None,
) -> Float[Tensor, "B ..."]:
    logpt = F.log_softmax(inputs, dim=1)
    inputs = (1 - logpt.exp()).pow(gamma) * logpt
    return F.nll_loss(inputs, target, weight, reduction='none')

    
class Model(pl.LightningModule):
    def __init__(
        self,

        # training parameters
        batch_size: int,
        seq_len: int,
        opt_args: dict[str, Any],
        focal_gamma: float,
        max_token_numel: int,

        # model hparams
        audio_encoder_ckpt: str,
        embed_dim: int,

        label_dim: int,
        label_emb_args: LabelEmbeddingArgs,

        decoder_args: DecoderArgs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # training params
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.opt_args = opt_args
        self.focal_gamma = focal_gamma
        self.max_token_numel = max_token_numel
        
        try:
            encode(Token(TokenType.TIMESTAMP, self.seq_len-1))
        except KeyError:
            raise ValueError('not enough timestamp tokens for sequence length- update `data/events.py`')

        # model
        audio_encoder = AudioEncoder.load_from_checkpoint(audio_encoder_ckpt)
        self.audio_encoder = audio_encoder.proj_audio
        self.a_dim = audio_encoder.a_dim

        self.embed = MP.Embedding(VOCAB_SIZE, embed_dim)
        token_head = MP.Linear(embed_dim, VOCAB_SIZE)
        token_head.weight = self.embed.weight
        self.token_head = nn.Sequential(
            token_head,
            MP.Gain(),
        )

        self.label_emb = LabelEmbedding(label_dim, label_emb_args)
        self.label_head = nn.Linear(embed_dim, NUM_LABELS)

        self.decoder = Decoder(
            embed_dim,
            self.a_dim,
            label_dim,
            decoder_args,
            self.seq_len,
        )

    def forward(
        self,
        labels: Float[Tensor, str(f"1 {NUM_LABELS}")],
        audio: Float[Tensor, str(f"1 {A_DIM} L")],
        tokens: Int[Tensor, "1 N"],
        timestamps: Float[Tensor, "1 N"],
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        
        D = audio.device
        features = self.audio_encoder(audio)[0].transpose(0,1) # L H
        b_feature_idxs, b_token_idxs, b_tokens = make_batch(
            tokens[0], timestamps[0],
            seq_len = self.seq_len,
            num_frames = features.size(0),
            max_batch_size = self.batch_size,
            max_token_numel = self.max_token_numel,
        )
        batch_size = b_feature_idxs.size(0)
        b_features = features[b_feature_idxs]

        # randomly mask labels for training
        b_labels = labels.repeat(batch_size, 1)
        label_embs = self.label_emb(th.where(th.rand_like(b_labels) < .5, -1, b_labels))

        b_prelude_tokens = th.tensor([DIFF, BOS], device=D).repeat(batch_size, 1)
        b_prelude_idxs = b_feature_idxs[:,:1].repeat(1, 2)
        h = self.decoder(
            x = self.embed(th.cat([b_prelude_tokens, b_tokens], dim=1)),
            x_t = th.cat([b_prelude_idxs, b_token_idxs], dim=1),
            ctx = b_features,
            ctx_t = b_feature_idxs,
            c = label_embs,
        ) # B N+2 E

        diff_emb, h = h[:,0], h[:,1:-1] # B E, B N E
        pred_labels = self.label_head(diff_emb) # B NUM_LABELS
        label_loss = (b_labels - pred_labels).pow(2).mean()

        pred_logits = self.token_head(h) # B N V
        token_loss = focal_loss(
            pred_logits.transpose(1,2),
            b_tokens,
            gamma = self.focal_gamma,
        ).mean()

        loss = token_loss + label_loss
        return loss, {
            "loss": loss.detach(),
            "token": token_loss.detach(),
            "label": label_loss.detach(),
            "b_tokens.numel": th.tensor(b_tokens.numel(), dtype=th.float),
        }
    
    @th.no_grad
    def sample(
        self,
        audio: Float[Tensor, str(f"{A_DIM} L")],
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
        time_budget: int | float = float('inf'), # max allowed time (sec)
    ) -> tuple[
        list[list[Token|float]],                # list of B lists of tokens and timestamps
        Float[Tensor, str(f"B {NUM_LABELS}")],  # predicted labels
    ]:
        from .sample import sample
        return sample(self, audio, labels, time_budget)


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