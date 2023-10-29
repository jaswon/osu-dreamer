
from jaxtyping import Float
from typing import Union

import numpy as np
from numpy import ndarray

from osu_dreamer.osu.beatmap import Beatmap
from osu_dreamer.osu.hit_objects import Slider, Spinner

from scipy.signal import find_peaks

from ..load_audio import FrameTimes

def hit_signal(bm: Beatmap, frame_times: FrameTimes) -> Float[ndarray, "4 L"]:
    """
    returns an array encoding a beatmap's hits:
    0. onsets
    1. sustains (both sliders and spinners)
    2. the first slide of sliders
    3. combos
    """

    return np.stack([
        onsets(bm, frame_times),
        extents([ (ho.t, ho.end_time())            for ho in bm.hit_objects if isinstance(ho, (Slider, Spinner)) ], frame_times),
        extents([ (ho.t, ho.t + ho.slide_duration) for ho in bm.hit_objects if isinstance(ho, Slider)            ], frame_times),
        extents(combo_regions(bm), frame_times),
    ])

def onsets(bm: Beatmap, frame_times: FrameTimes) -> Float[ndarray, "L"]:
    """returns time (in log ms) since last onset, scaled+shifted to [0,1]"""
    MIN_TIME = 4    # intervals shorter than 2^MIN_TIME milliseconds get aliased to 0
    MAX_TIME = 10   # intervals longer  than 2^MAX_TIME milliseconds get aliased to 1

    onsets = np.full_like(frame_times, 2**MAX_TIME)
    for ho in bm.hit_objects:
        hit = frame_times - ho.t
        region = (hit >= 0) & (hit <= 2**MAX_TIME)
        onsets[region] = hit[region]

    log_onsets = np.log2(onsets + 2**MIN_TIME).clip(MIN_TIME, MAX_TIME)
    return (log_onsets - MIN_TIME) / (MAX_TIME - MIN_TIME)

def decode_onsets(onsets: Float[ndarray, "L"]) -> list[int]:
    return find_peaks(-onsets, height=0.6, distance=4)[0].tolist()

Real = Union[int, float]

def combo_regions(bm: Beatmap) -> list[tuple[Real, Real]]:
    new_combo_regions = []
    region_end = None
    for ho in bm.hit_objects[::-1]:
        if region_end is None:
            region_end = ho.end_time() + 1
        if ho.new_combo:
            new_combo_regions.insert(0, (ho.t, region_end))
            region_end = None

    return new_combo_regions

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