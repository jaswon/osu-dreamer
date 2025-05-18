
from typing import Union, Iterator

from jaxtyping import UInt, Float
import numpy as np

from osu_dreamer.osu.beatmap import Beatmap
from osu_dreamer.osu.hit_objects import Circle, HitObject, Slider, Spinner
from osu_dreamer.osu.sliders import Bezier, Line, Perfect

from .events import EventType, Event, encode

def location_event(x: int, y: int) -> Event:
    r = 4 if (0<=x<=512) and (0<=y<=384) else 16
    return Event(EventType.LOCATION, (round(x/r)*r, round(y/r)*r))

def onset_events(ho: HitObject) -> Iterator[Event]:
    yield Event(EventType.ONSET, (ho.new_combo, ho.whistle, ho.finish, ho.clap))

def slider_events(ho: Slider) -> Iterator[Event]:
    yield Event(EventType.SLIDES, ho.slides)

    match ho:
        case Line(ctrl_pts=[a,b]):
            yield location_event(*a)
            yield Event(EventType.LINE)
            yield location_event(*b)
        case Perfect(ctrl_pts=[a,b,c]):
            yield location_event(*a)
            yield Event(EventType.PERFECT)
            yield location_event(*b)
            yield location_event(*c)
        case Bezier(ctrl_pts=[a,b,*rest]):
            yield location_event(*a)
            yield Event(EventType.BEZIER)
            yield location_event(*b)
            last = tuple(b)
            for c in rest:
                if tuple(c) == last:
                    yield Event(EventType.KNOT)
                else:
                    yield location_event(*c)
                last = tuple(c)
            yield Event(EventType.BEZIER_END)
        case _:
            raise ValueError(f'unexpected slider type and control points: ({type(ho)}) {ho.ctrl_pts}')

def beatmap_events(bm: Beatmap) -> Iterator[Union[Event, int]]:
    breaks = iter(bm.breaks)
    next_break = next(breaks, None)
    for ho in bm.hit_objects:
        if next_break is not None and next_break.t < ho.t:
            yield next_break.t
            yield Event(EventType.BREAK)

            yield next_break.end_time()
            yield Event(EventType.RELEASE)
            next_break = next(breaks, None)

        yield ho.t
        match ho:
            case Circle(): yield Event(EventType.CIRCLE)
            case Spinner(): yield Event(EventType.SPINNER)
            case Slider(): yield Event(EventType.SLIDER)
        yield from onset_events(ho)
        if isinstance(ho, Circle):
            yield location_event(ho.x, ho.y)
        else:
            yield ho.end_time()
            yield Event(EventType.RELEASE)
            if isinstance(ho, Slider):
                try:
                    yield from slider_events(ho)
                except Exception as e:
                    raise ValueError(f'bad slider @ {ho.t} in {bm.filename}') from e

def to_tokens_and_timestamps(bm: Beatmap) -> tuple[
    UInt[np.ndarray, "N"],  # tokens
    Float[np.ndarray, "N"],  # timestamps
]:
    """
    return tokens, timestamps, and ranges for each event
    """
    tokens: list[int] = []
    timestamps: list[float] = []
    cur_t: float
    for i, event in enumerate(beatmap_events(bm)):
        match event:
            case int(t):
                cur_t = float(t)
            case Event():
                tokens.append(encode(event))
                timestamps.append(cur_t)
    return (
        np.array(tokens, dtype=np.uint), 
        np.array(timestamps, dtype=float), 
    )