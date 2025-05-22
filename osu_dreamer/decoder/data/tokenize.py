
from typing import Iterator

from dataclasses import dataclass

from jaxtyping import Int, Float
import numpy as np

from osu_dreamer.osu.beatmap import Beatmap
from osu_dreamer.osu.hit_objects import Circle, HitObject, Slider, Spinner
from osu_dreamer.osu.sliders import Bezier, Line, Perfect

from .tokens import TokenType, Token, encode, PAD

@dataclass
class TimingToken:
    t: float

    def __str__(self):
        return f"t= {self.t:.2f}"

@dataclass
class PositionToken:
    x: float
    y: float

    def __str__(self):
        return f"({self.x:.2f}, {self.y:.2f})"

def timing_token(t: int|float) -> TimingToken:
    return TimingToken(float(t))

def location_token(x: int|float, y: int|float) -> PositionToken:
    # rescale (0,512)x(0,384) -> (-4,4)x(-3,3)
    return PositionToken(float(x)/64-4, float(y)/64-3)

def onset_tokens(ho: HitObject) -> Iterator[Token]:
    yield Token(TokenType.FLAGS, (ho.new_combo, ho.whistle, ho.finish, ho.clap))

def slider_tokens(ho: Slider) -> Iterator[Token | TimingToken | PositionToken]:
    yield Token(TokenType.SLIDES, min(99,ho.slides))

    match ho:
        case Line(ctrl_pts=[a,b]):
            yield location_token(*a)
            yield Token(TokenType.LINE)
            yield location_token(*b)
        case Perfect(ctrl_pts=[a,b,c]):
            yield location_token(*a)
            yield Token(TokenType.PERFECT)
            yield location_token(*b)
            yield location_token(*c)
        case Bezier(ctrl_pts=[a,b,*rest]):
            yield location_token(*a)
            yield Token(TokenType.BEZIER)
            yield location_token(*b)
            last = tuple(b)
            for c in rest:
                if tuple(c) == last:
                    yield Token(TokenType.KNOT)
                else:
                    yield location_token(*c)
                last = tuple(c)
            yield Token(TokenType.BEZIER_END)
        case _:
            raise ValueError(f'unexpected slider type and control points: ({type(ho)}) {ho.ctrl_pts}')

def beatmap_tokens(bm: Beatmap) -> Iterator[Token | TimingToken | PositionToken]:
    breaks = iter(bm.breaks)
    next_break = next(breaks, None)
    for ho in bm.hit_objects:
        if next_break is not None and next_break.t < ho.t:
            yield timing_token(next_break.t)
            yield Token(TokenType.BREAK)

            yield timing_token(next_break.end_time())
            yield Token(TokenType.RELEASE)
            next_break = next(breaks, None)

        yield timing_token(ho.t)
        match ho:
            case Circle(): yield Token(TokenType.CIRCLE)
            case Spinner(): yield Token(TokenType.SPINNER)
            case Slider(): yield Token(TokenType.SLIDER)
        yield from onset_tokens(ho)
        if isinstance(ho, Circle):
            yield location_token(ho.x, ho.y)
        else:
            yield timing_token(ho.end_time())
            yield Token(TokenType.RELEASE)
            if isinstance(ho, Slider):
                try:
                    yield from slider_tokens(ho)
                except Exception as e:
                    raise ValueError(f'bad slider @ {ho.t} in {bm.filename}') from e

def tokenize(bm: Beatmap) -> tuple[
    Int[np.ndarray, "N"],       # type
    Int[np.ndarray, "N"],       # tokens
    Float[np.ndarray, "N"],     # timestamps
    Float[np.ndarray, "N 2"],   # positions
]:
    """
    return (types, tokens, timestamps, positions) for the beatmap
    """
    types: list[int] = []
    tokens: list[int] = []
    timestamps: list[float] = []
    positions: list[tuple[float,float]] = []
    cur_t: float
    cur_p = (256.,192.)
    for i, token in enumerate(beatmap_tokens(bm)):
        match token:
            case Token():
                typ = 0
                token_id = encode(token)
            case TimingToken(t):
                typ = 1
                token_id = PAD
                cur_t = t
            case PositionToken(x,y):
                typ = 2
                token_id = PAD
                cur_p = (x,y)
        types.append(typ)
        tokens.append(token_id)
        timestamps.append(cur_t)
        positions.append(cur_p)
    return (
        np.array(types, dtype=int),
        np.array(tokens, dtype=int), 
        np.array(timestamps, dtype=float), 
        np.array(positions, dtype=float),
    )