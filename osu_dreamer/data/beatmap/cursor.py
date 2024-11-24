
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

    hos: list[HitObject] = [Circle(0, True, 0, 256, 192), *bm.hit_objects]

    sig = np.zeros((frame_times.shape[0], 2))

    for cur, nxt in zip(hos, hos[1:] + [None]):

        # hit current object
        cur_i = (frame_times >= cur.t) & (frame_times < cur.end_time())
        if isinstance(cur, Spinner):
            sig[cur_i] = cur.start_pos()
        elif isinstance(cur, Slider):
            cur_f = (frame_times[cur_i] - cur.t) % (cur.slide_duration * 2) / cur.slide_duration
            sig[cur_i] = cur.lerp(np.where(cur_f < 1, cur_f, 2 - cur_f))

        if nxt is None:
            # end of map
            sig[frame_times >= cur.end_time()] = cur.end_pos()
            break

        # current object hit, wait for next object to appear
        wait_t = (frame_times >= cur.end_time()) & (frame_times < nxt.t - preempt)
        sig[wait_t] = cur.end_pos()

        # approach next hit object
        approach_start_time = max(cur.end_time(), nxt.t - preempt)
        approach_i = (frame_times >= approach_start_time) & (frame_times < nxt.t)
        approach_f = (frame_times[approach_i] - approach_start_time) / (nxt.t - approach_start_time)
        sig[approach_i] = (1 - approach_f[:,None]) * cur.end_pos() + approach_f[:,None] * nxt.start_pos()
        
    return (np.array(sig) / np.array([256,192])).T - 1
