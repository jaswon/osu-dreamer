
from jaxtyping import Float

import warnings

import numpy as np
from numpy import ndarray

from osu_dreamer.osu.beatmap import Beatmap
from osu_dreamer.osu.hit_objects import Circle, Slider, Spinner, HitObject

from ..load_audio import FrameTimes

from enum import IntEnum
CursorEncoding = IntEnum('CursorEncoding', [ "X", "Y" ], start=0)
CURSOR_DIM = len(CursorEncoding)

CursorSignal = Float[ndarray, str(f"{CURSOR_DIM} L")]

def cursor_signal(bm: Beatmap, frame_times: FrameTimes) -> CursorSignal:
    """
    encodes the position of the cursor at `frame_times` (ms)

    (0,1) -- (1,1)
      |        |
    (0,0) -- (1,0)
    """
    if len(bm.hit_objects) == 0:
        warnings.warn('beatmap has no hit objects')

    # compute duration a hit object stays on screen from approach rate
    preempt = 1200 + (120 if bm.ar <= 5 else 150) * (5-bm.ar)

    start = Circle(0, True, 256, 192)
    hos: list[HitObject] = [start, *bm.hit_objects]

    pos: list[Float[ndarray, "2"]] = []

    for cur, nxt in zip(hos, hos[1:] + [None]):

        # hit current object
        if isinstance(cur, Spinner):
            cur_count = np.sum((frame_times >= cur.t) & (frame_times < cur.end_time()))
            pos.extend(cur.start_pos()[None].repeat(cur_count, axis=0))
        elif isinstance(cur, Slider):
            cur_t = frame_times[(frame_times >= cur.t) & (frame_times < cur.end_time())]
            cur_f = (cur_t - cur.t) % (cur.slide_duration * 2) / cur.slide_duration
            pos.extend(cur.lerp(np.where(cur_f < 1, cur_f, 2 - cur_f)))

        if nxt is None:
            # end of map
            map_end_count = np.sum(frame_times >= cur.end_time())
            pos.extend(cur.end_pos()[None].repeat(map_end_count, axis=0))
            break

        # current object hit, wait for next object to appear
        wait_count = np.sum((frame_times >= cur.end_time()) & (frame_times < nxt.t - preempt))
        pos.extend(cur.end_pos()[None].repeat(wait_count, axis=0))

        # approach next hit object
        start_time = max(cur.end_time(), nxt.t - preempt)
        approach_t = frame_times[(frame_times >= start_time) & (frame_times < nxt.t)]
        approach_f = (approach_t - start_time) / (nxt.t - start_time)
        pos.extend((1 - approach_f[:,None]) * cur.end_pos() + approach_f[:,None] * nxt.start_pos())
        
    return (np.array(pos) / np.array([256,192])).T - 1
