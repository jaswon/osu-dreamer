
from jaxtyping import Float
from typing import Optional

import warnings

import numpy as np
from numpy import ndarray

from osu_dreamer.osu.beatmap import Beatmap
from osu_dreamer.osu.hit_objects import Circle, Slider, Spinner, HitObject

from ..load_audio import FrameTimes

def preempt(ar: float) -> float:
    """compute duration a hit object stays on screen from Approach Rate (AR)"""
    factor = 120 if ar <= 5 else 150
    return 1200 + factor * (5-ar)

def cursor_signal(bm: Beatmap, frame_times: FrameTimes) -> Float[ndarray, "2 L"]:
    """
    encodes the position of the cursor at `frame_times` (ms)

    (0,1) -- (1,1)
      |        |
    (0,0) -- (1,0)
    """
    hos = iter(bm.hit_objects)
    cur: HitObject = Circle(0, False, 256, 192)
    nxt: Optional[HitObject] = next(hos, None)
    if nxt is None:
        warnings.warn('beatmap has no hit objects')

    pos: list[Float[ndarray, "2"]] = []

    bm_preempt = preempt(bm.ar)

    for t in frame_times:

        # update cur, nxt
        while nxt is not None and nxt.t <= t:
            cur, nxt = nxt, next(hos, None)

        if t < cur.end_time():
            # hitting current hit object
            # note: will not be a `Circle` by definition of `cur`
            if isinstance(cur, Spinner):
                pos.append(cur.start_pos())
            elif isinstance(cur, Slider):
                ts = (t - cur.t) % (cur.slide_duration * 2) / cur.slide_duration
                if ts < 1:  # start -> end
                    pos.append(cur.lerp(ts))
                else:  # end -> start
                    pos.append(cur.lerp(2 - ts))
        elif nxt is None:
            # after last hit object
            pos.append(cur.end_pos())
        else:
            # current object completed, look for next object
            if nxt.t - t > bm_preempt:
                # next object hasn't appeared yet, don't approach
                pos.append(cur.end_pos())
            else:
                # move (lerp) to next hit object
                start_time = max(cur.end_time(), nxt.t - bm_preempt)
                f = (t - start_time) / (nxt.t - start_time)
                pos.append((1 - f) * cur.end_pos() + f * nxt.start_pos())
        
    return (np.array(pos) / np.array([512,384])).T