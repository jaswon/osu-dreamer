
from typing import Iterator

from dataclasses import dataclass

from jaxtyping import Int, Float
import numpy as np

from osu_dreamer.osu.beatmap import Beatmap
from osu_dreamer.osu.hit_objects import Circle, Slider, Spinner
from osu_dreamer.osu.sliders import Bezier, Line, Perfect

from .tokens import BOS, EOS, TokenType, Token, encode, PAD

@dataclass
class TimingToken:
    t: float

    def __str__(self):
        return f"TIMESTAMP({self.t:.2f})"

@dataclass
class PositionToken:
    x: float
    y: float

    def __str__(self):
        return f"POSITION({self.x:.2f}, {self.y:.2f})"

def slider_tokens(ho: Slider) -> Iterator[Token | PositionToken]:
    yield Token(TokenType.SLIDES, min(99,ho.slides))

    match ho:
        case Line(ctrl_pts=[a,b]):
            yield PositionToken(*a)
            yield Token(TokenType.LINE)
            yield PositionToken(*b)
        case Perfect(ctrl_pts=[a,b,c]):
            yield PositionToken(*a)
            yield Token(TokenType.PERFECT)
            yield PositionToken(*b)
            yield PositionToken(*c)
        case Bezier(ctrl_pts=[a,b,*rest]):
            yield PositionToken(*a)
            yield Token(TokenType.BEZIER)
            yield PositionToken(*b)
            last = tuple(b)
            for c in rest:
                if tuple(c) == last:
                    yield Token(TokenType.KNOT)
                else:
                    yield PositionToken(*c)
                last = tuple(c)
            yield Token(TokenType.BEZIER_END)
        case _:
            raise ValueError(f'unexpected slider type and control points: ({type(ho)}) {ho.ctrl_pts}')

def beatmap_tokens(bm: Beatmap) -> Iterator[Token | TimingToken | PositionToken]:
    breaks = iter(bm.breaks)
    next_break = next(breaks, None)
    for ho in bm.hit_objects:
        if next_break is not None and next_break.t < ho.t:
            yield TimingToken(float(next_break.t))
            yield Token(TokenType.BREAK)

            yield TimingToken(float(next_break.end_time()))
            yield Token(TokenType.RELEASE)
            next_break = next(breaks, None)

        yield TimingToken(float(ho.t))
        match ho:
            case Circle(): yield Token(TokenType.CIRCLE)
            case Spinner(): yield Token(TokenType.SPINNER)
            case Slider(): yield Token(TokenType.SLIDER)
        yield Token(TokenType.FLAGS, (ho.new_combo, ho.whistle, ho.finish, ho.clap))
        if isinstance(ho, Circle):
            yield PositionToken(ho.x, ho.y)
        else:
            yield TimingToken(float(ho.end_time()))
            yield Token(TokenType.RELEASE)
            if isinstance(ho, Slider):
                try:
                    yield from slider_tokens(ho)
                except Exception as e:
                    raise ValueError(f'bad slider @ {ho.t} in {bm.filename}') from e

def tokenize(bm: Beatmap) -> tuple[
    Int[np.ndarray, "N"],       # modes
    Int[np.ndarray, "N"],       # tokens
    Float[np.ndarray, "N"],     # timestamps
    Float[np.ndarray, "N 2"],   # positions
]:
    """
    return (modes, tokens, timestamps, positions) for the beatmap
    """
    modes: list[int] = [0]
    tokens: list[int] = [BOS]
    timestamps: list[float] = [0]
    positions: list[tuple[float,float]] = [(0,0)]
    cur_t: float
    cur_p = (0.,0.)
    for i, token in enumerate(beatmap_tokens(bm)):
        match token:
            case Token():
                mode = 0
                token_id = encode(token)
            case TimingToken(t):
                mode = 1
                token_id = PAD
                cur_t = t
            case PositionToken(x,y):
                mode = 2
                token_id = PAD
                # rescale (0,512)x(0,384) -> (-4,4)x(-3,3)
                cur_p = (float(x)/64-4, float(y)/64-3)
        modes.append(mode)
        tokens.append(token_id)
        timestamps.append(cur_t)
        positions.append(cur_p)
    modes.append(0)
    tokens.append(EOS)
    timestamps.append(cur_t)
    positions.append(cur_p)
    return (
        np.array(modes, dtype=int),
        np.array(tokens, dtype=int), 
        np.array(timestamps, dtype=float), 
        np.array(positions, dtype=float),
    )