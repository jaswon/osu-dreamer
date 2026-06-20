
from typing import BinaryIO
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
]
HIT_DIM = len(HitSignals)

CursorSignals = [
    BeatmapEncoding.X,
    BeatmapEncoding.Y,
]
CURSOR_DIM = len(CursorSignals)

EncodedBeatmap = Float[ndarray, str(f"{X_DIM} L")]

NUM_LABELS = 5
Labels = Float[np.ndarray, f'{NUM_LABELS}']

def get_labels(bm: Beatmap) -> Labels:
    return np.array([bm.sr, bm.ar, bm.od, bm.cs, bm.hp])

### DISK FORMAT

HIT_DTYPE = np.uint8
XY_DTYPE = np.uint16

def write_beatmap(f: BinaryIO, bm: Beatmap, frame_times: FrameTimes):
    hit = hit_signal(bm, frame_times)
    xy = cursor_signal(bm, frame_times)
    xy_min = xy.min(axis=1, keepdims=True)
    xy_rng = xy.max(axis=1, keepdims=True) - xy_min
    xy_rng[xy_rng == 0.] = 1.
    xy_norm = (xy - xy_min) / xy_rng
    np.savez(f, allow_pickle=False,
        hit = np.round(hit * np.iinfo(HIT_DTYPE).max).astype(HIT_DTYPE),
        xy = np.round(xy_norm * np.iinfo(XY_DTYPE).max).astype(XY_DTYPE),
        xy_min = xy_min,
        xy_rng = xy_rng,
        labels = get_labels(bm),
    )

def read_beatmap(f: BinaryIO) -> tuple[EncodedBeatmap, Labels]:
    with np.load(f) as npz:
        hit, xy, xy_min, xy_rng, labels = npz['hit'], npz['xy'], npz['xy_min'], npz['xy_rng'], npz['labels']
    return np.concatenate([
        hit.astype(float) / np.iinfo(HIT_DTYPE).max,
        xy.astype(float) / np.iinfo(XY_DTYPE).max * xy_rng + xy_min ,
    ]), labels