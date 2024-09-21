
from jaxtyping import Float
from typing import Union

import numpy as np
from numpy import ndarray

from osu_dreamer.osu.beatmap import Beatmap
from osu_dreamer.osu.hit_objects import Slider, Spinner

from scipy.signal import find_peaks

from ..load_audio import FrameTimes

from enum import IntEnum
HitEncoding = IntEnum('HitEncoding', [
    "ONSET",
    "SUSTAIN",
    "SLIDER",
    "COMBO",
], start=0)
HIT_DIM = len(HitEncoding)

HitSignal = Float[ndarray, str(f"{HIT_DIM} L")]

def hit_signal(bm: Beatmap, frame_times: FrameTimes) -> HitSignal:
    """
    returns an array encoding a beatmap's hits:
    0. onsets
    1. new combos
    2. sustains (both sliders and spinners)
    3. the first slide of sliders
    """

    return np.stack([
        events(onsets(bm), frame_times),
        events(new_combos(bm), frame_times),
        extents([ (ho.t, ho.end_time())            for ho in bm.hit_objects if isinstance(ho, (Slider, Spinner)) ], frame_times),
        extents([ (ho.t, ho.t + ho.slide_duration) for ho in bm.hit_objects if isinstance(ho, Slider)            ], frame_times),
    ]) * 2 - 1

Real = Union[int, float]

def onsets(bm: Beatmap) -> list[Real]:
    return [ ho.t for ho in bm.hit_objects]

def new_combos(bm: Beatmap) -> list[Real]:
    return [ ho.t for ho in bm.hit_objects if ho.new_combo ]

def events(ts: list[Real], frame_times: FrameTimes) -> Float[ndarray, "L"]:
    """returns time (in log ms) since last event, scaled+shifted to [0,1]"""
    MIN_TIME = 4    # intervals shorter than 2^MIN_TIME milliseconds get aliased to 0
    MAX_TIME = 11   # intervals longer  than 2^MAX_TIME milliseconds get aliased to 1

    time_since_last_event = np.full_like(frame_times, 2**MAX_TIME)
    for t in ts:
        time_since_event = frame_times - t
        region = (time_since_event >= 0) & (time_since_event <= 2**MAX_TIME)
        time_since_last_event[region] = time_since_event[region]

    log_time = np.log2(time_since_last_event + 2**MIN_TIME).clip(MIN_TIME, MAX_TIME)
    return (log_time - MIN_TIME) / (MAX_TIME - MIN_TIME)

def decode_events(events: Float[ndarray, "L"]) -> list[int]:
    return find_peaks(-events, height=0.6, distance=4)[0].tolist()

def extents(regions: list[tuple[Real, Real]], frame_times: FrameTimes) -> Float[ndarray, "L"]:
    """1 within extents, 0 everywhere else"""
    holds = np.zeros_like(frame_times)
    for s, e in regions:
        holds[(frame_times >= s) & (frame_times < e)] = 1
    return holds

def decode_extents(extents: Float[ndarray, "L"]) -> tuple[list[int], list[int]]:
    before_below = extents[:-1] <= 0
    after_below  = extents[1:]  <= 0

    start_idxs = sorted(np.argwhere(before_below & ~after_below)[:,0].tolist())
    end_idxs   = sorted(np.argwhere(~before_below & after_below)[:,0].tolist())

    # ensure that start_idxs[i] < end_idxs[i] for all 0 <= i < min(len(start_idxs), len(end_idxs))
    cursor = 0
    for cursor, start in enumerate(start_idxs):
        try:
            while start >= end_idxs[cursor]:
                end_idxs.pop(cursor)
        except IndexError:
            break
    cursor += 1

    return start_idxs[:cursor], end_idxs[:cursor]