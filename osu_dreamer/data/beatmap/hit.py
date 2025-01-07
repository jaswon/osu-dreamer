
from jaxtyping import Float
from typing import Union

import numpy as np
from numpy import ndarray

from osu_dreamer.osu.beatmap import Beatmap
from osu_dreamer.osu.hit_objects import Slider, Spinner

from ..load_audio import FrameTimes

from enum import IntEnum

Real = Union[int, float]

# == events ==

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
    events_pp = events[2:] + events[:-2] - 2 * events[1:-1]
    return (np.nonzero(events_pp > .5)[0] + 1).tolist()

# == extents ==

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

# == hit signal ==

HitEncoding = IntEnum('HitEncoding', [
    "ONSET",
    "COMBO",
    "SLIDE",
    "SUSTAIN",
    "WHISTLE",
    "FINISH",
    "CLAP",
], start=0)
HIT_DIM = len(HitEncoding)

HitSignal = Float[ndarray, str(f"{HIT_DIM} L")]

def hit_signal(bm: Beatmap, frame_times: FrameTimes) -> HitSignal:
    """
    returns an array encoding a beatmap's hits:
    0. onsets
    1. new combos
    2. slide ends
    3. sustains (both sliders and spinners)
    4. whistle hit sound
    5. finish hit sound
    6. clap hit sound
    """

    slide_times = []
    for ho in bm.hit_objects:
        if not isinstance(ho, Slider):
            continue
        slide_times.extend([ ho.t + ho.slide_duration * (i+1) for i in range(ho.slides) ])

    return np.stack([
        events([ ho.t for ho in bm.hit_objects                 ], frame_times), # onsets
        events([ ho.t for ho in bm.hit_objects if ho.new_combo ], frame_times), # new combos
        events(slide_times, frame_times), # slides
        extents([
            (ho.t, ho.end_time())
            for ho in bm.hit_objects
            if isinstance(ho, (Slider, Spinner))
        ], frame_times), # sustains
        events([ ho.t for ho in bm.hit_objects if ho.whistle ], frame_times), # whistles
        events([ ho.t for ho in bm.hit_objects if ho.finish ], frame_times), # finishes
        events([ ho.t for ho in bm.hit_objects if ho.clap ], frame_times), # claps
    ]) * 2 - 1

Hit = Union[
    tuple[int, bool, bool, bool, bool],           #  hit(t, new_combo, whistle, finish, clap)
    tuple[int, bool, bool, bool, bool, int, int], # hold(t, new_combo, whistle, finish, clap, u, slides)
]

ONSET_TOL = 2
def decode_hit_signal(hit_signal: HitSignal) -> list[Hit]:
    onsets = hit_signal[HitEncoding.ONSET]
    onset_idxs = decode_events(onsets)

    # maps signal index to onset
    onset_idx_map = np.full_like(onsets, -1, dtype=int)
    for i, onset_idx in enumerate(onset_idxs):
        onset_idx_map[onset_idx-ONSET_TOL:onset_idx+ONSET_TOL+1] = i

    onset_props = np.full((len(onset_idxs), 4), False, dtype=bool)
    for i, sig in enumerate([HitEncoding.COMBO, HitEncoding.WHISTLE, HitEncoding.FINISH, HitEncoding.CLAP]):
        for ev in decode_events(hit_signal[sig]):
            onset_idx = onset_idx_map[ev]
            if onset_idx == -1:
                continue
            onset_props[onset_idx, i] = True

    sustain_ends = [-1] * len(onset_idxs)
    for sustain_start, sustain_end in zip(*decode_extents(hit_signal[HitEncoding.SUSTAIN])):
        onset_idx = onset_idx_map[sustain_start]
        if onset_idx == -1:
            continue
        sustain_ends[onset_idx] = sustain_end

    slides = np.zeros_like(hit_signal[0])
    slides[decode_events(hit_signal[HitEncoding.SLIDE])] = 1

    hits: list[Hit] = []
    for onset_loc, onset_prop, sustain_end in zip(onset_idxs, onset_props, sustain_ends):
        hit = (onset_loc, *onset_prop.tolist())

        if sustain_end == -1 or sustain_end - onset_loc < 4:
            # sustain too short
            hits.append(hit)
            continue

        num_slides = int(slides[onset_loc:sustain_end+1].sum())
        hits.append((*hit, sustain_end, num_slides))
    
    return hits