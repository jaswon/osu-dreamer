
from typing import Union
from jaxtyping import UInt

from enum import IntEnum, auto

import numpy as np

from osu_dreamer.data.load_audio import FrameTimes
from osu_dreamer.osu.beatmap import Beatmap
from osu_dreamer.osu.hit_objects import Circle, Slider, Spinner

Real = Union[int, float]

class Event(IntEnum):
    NULL = 0 # no event
    CIRCLE = auto()
    SLIDER = auto()
    SPINNER = auto()
    BREAK = auto()

NUM_EVENTS = len(Event)

def beatmap_events(bm: Beatmap, frame_times: FrameTimes) -> UInt[np.ndarray, "L"]:

    events = np.full(len(frame_times), Event.NULL, dtype=np.uint8)

    for b in bm.breaks:
        events[(frame_times >= b.t) & (frame_times < b.end_time())] = Event.BREAK

    for ho in bm.hit_objects:
        if isinstance(ho, Circle):
            events[np.argmin(abs(frame_times - ho.t))] = Event.CIRCLE
        elif isinstance(ho, Spinner):
            events[(frame_times >= ho.t) & (frame_times < ho.end_time())] = Event.SPINNER
        elif isinstance(ho, Slider):
            events[(frame_times >= ho.t) & (frame_times < ho.end_time())] = Event.SLIDER
            
    return events