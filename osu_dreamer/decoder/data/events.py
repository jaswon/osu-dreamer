
from dataclasses import dataclass

@dataclass(frozen=True)
class Event:
    t: int

@dataclass(frozen=True)
class Sustain(Event):
    u: int

@dataclass(frozen=True)
class Break(Sustain): pass

@dataclass(frozen=True)
class Onset(Event):
    new_combo: bool
    whistle: bool
    finish: bool
    clap: bool

Position = tuple[int,int]

@dataclass(frozen=True)
class Circle(Onset):
    pos: Position

@dataclass(frozen=True)
class Spinner(Onset, Sustain): pass

@dataclass(frozen=True)
class Slider(Onset, Sustain):
    slides: int
    control_points: list[Position]

@dataclass(frozen=True)
class Line(Slider): pass

@dataclass(frozen=True)
class Perfect(Slider): pass

@dataclass(frozen=True)
class Bezier(Slider): pass