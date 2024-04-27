
from jaxtyping import Float

import numpy as np
from numpy import ndarray

from osu_dreamer.osu.beatmap import Beatmap

from ..load_audio import FrameTimes

from enum import IntEnum
TimingEncoding = IntEnum('TimingEncoding', [
    "BEAT_PHASE",
    "MEASURE_PHASE",
], start=0)
TIMING_DIM = len(TimingEncoding)

TimingSignal = Float[ndarray, str(f"{TIMING_DIM} L")]

def timing_signal(bm: Beatmap, frame_times: FrameTimes) -> TimingSignal:
    """
    returns an array encoding a song's timing:
    0. beat phase
    1. measure phase
    """
    sig = np.zeros((2, len(frame_times)))

    for i, tp in enumerate(bm.uninherited_timing_points):
        start = tp.t
        if i == 0:
            # rewind start to beginning of song
            measure_length = tp.beat_length * tp.meter
            start -= (start // measure_length + 1) * measure_length
        window = frame_times >= start
        beat_phase = ((frame_times - start) / tp.beat_length)
        measure_phase = beat_phase / tp.meter
        sig[0, window] = beat_phase[window] % 1
        sig[1, window] = measure_phase[window] % 1

    return sig