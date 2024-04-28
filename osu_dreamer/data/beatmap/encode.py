
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
    "SUSTAIN",
    "SLIDER",
    "COMBO",

    # cursor signals
    "CURSOR_X",
    "CURSOR_Y",
], start=0)
X_DIM = len(BeatmapEncoding)

CURSOR_SIGNALS = [
    BeatmapEncoding.CURSOR_X,
    BeatmapEncoding.CURSOR_Y,
]
CURSOR_DIM = len(CURSOR_SIGNALS)

EncodedBeatmap = Float[ndarray, str(f"{X_DIM} L")]

def encode_beatmap(bm: Beatmap, frame_times: FrameTimes) -> EncodedBeatmap:

    return np.concatenate([
        hit_signal(bm, frame_times),
        cursor_signal(bm, frame_times),
    ], axis=0) * 2 - 1 # [0,1] -> [-1,1]