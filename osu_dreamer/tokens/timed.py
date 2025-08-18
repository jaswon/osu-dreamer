
from dataclasses import dataclass
from typing import Union

Coordinate = tuple[int, int]

@dataclass
class BeatLen:
    """uninherited timing point"""
    ms: float
    meter: int

@dataclass
class SliderMult:
    """inherited timing point"""
    mult: float

@dataclass
class Break:
    duration: int

@dataclass
class HitObject:
    new_combo: bool
    whistle: bool
    finish: bool
    clap: bool

@dataclass
class HitCircle(HitObject):
    p: Coordinate

@dataclass
class _Hold(HitObject):
    duration: int

@dataclass
class Spinner(_Hold):
    pass

@dataclass
class _Slider(_Hold):
    slides: int

@dataclass
class PerfectSlider(_Slider):
    p: Coordinate
    q: Coordinate

    # angular deviation from straight line to end point (-pi, pi)
    # `deviation` > 0 => arc deviates to the left
    # `deviation` = 0 => line slider
    deviation: float

@dataclass
class BezierSlider(_Slider):
    shape: list[Coordinate]


Timed = Union[
    BeatLen,
    SliderMult,
    Break,
    HitCircle,
    Spinner,
    PerfectSlider,
    BezierSlider,
]