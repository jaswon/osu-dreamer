
from jaxtyping import Float

import numpy as np
from numpy import ndarray

from osu_dreamer.osu.beatmap import Beatmap
from ..load_audio import FrameTimes

from .cursor import cursor_signal
from .hit import hit_signal

from enum import IntEnum
BeatmapEncoding = IntEnum('BeatmapEncoding', [
    # hit signals
    "ONSET",
    "COMBO",
    "SLIDE",
    "SUSTAIN",
    "WHISTLE",
    "FINISH",
    "CLAP",
    "BEAT",
    "METER",

    # cursor signals
    "X",
    "Y",
], start=0)
X_DIM = len(BeatmapEncoding)

HitSignals = [
    BeatmapEncoding.ONSET,
    BeatmapEncoding.COMBO,
    BeatmapEncoding.SLIDE,
    BeatmapEncoding.SUSTAIN,
    BeatmapEncoding.WHISTLE,
    BeatmapEncoding.FINISH,
    BeatmapEncoding.CLAP,
    BeatmapEncoding.BEAT,
    BeatmapEncoding.METER,
]
HIT_DIM = len(HitSignals)

CursorSignals = [
    BeatmapEncoding.X,
    BeatmapEncoding.Y,
]
CURSOR_DIM = len(CursorSignals)

EncodedBeatmap = Float[ndarray, str(f"{X_DIM} L")]

def encode_beatmap(bm: Beatmap, frame_times: FrameTimes) -> EncodedBeatmap:
    return np.concatenate([
        hit_signal(bm, frame_times),
        cursor_signal(bm, frame_times),
    ], axis=0)