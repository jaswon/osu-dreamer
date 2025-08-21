
from dataclasses import dataclass
from typing import Union

Coordinate = tuple[int, int]

@dataclass
class BeatLen:
    """uninherited timing point"""
    ms: float
    
    def __str__(self):
        return f"BeatLen({self.ms})"

@dataclass
class SliderVel:
    """inherited timing point"""
    vel: float
    
    def __str__(self):
        return f"SliderVel({self.vel})"

@dataclass
class Break:
    duration: int
    
    def __str__(self):
        return f"Break[duration={self.duration}]"

@dataclass
class HitObject:
    new_combo: bool
    whistle: bool
    finish: bool
    clap: bool

    def _cls_str(self) -> str:
        return self.__class__.__name__

    def _object_str(self) -> str:
        return ""
    
    def __str__(self) -> str:
        return "".join([
            "*" if self.new_combo else " ",
            self._cls_str(),
            f"[{"".join([
                "w" if self.whistle else "_",
                "f" if self.finish else "_",
                "c" if self.clap else "_",
            ])}]",
            self._object_str(),
        ])

@dataclass
class HitCircle(HitObject):
    p: Coordinate

    def _cls_str(self) -> str:
        return "C"

    def _object_str(self) -> str:
        return f" {self.p}"

@dataclass
class Hold(HitObject):
    duration: int

    def _hold_str(self) -> str:
        return ""

    def _object_str(self) -> str:
        return f"[duration={self.duration}]{self._hold_str()}"

@dataclass
class Spinner(Hold):
    def _cls_str(self) -> str:
        return "S"

@dataclass
class Slider(Hold):
    slides: int
    head: Coordinate

    def _slider_str(self) -> str:
        return ""

    def _hold_str(self):
        return f"[slides={self.slides}]{self._slider_str()}"

@dataclass
class PerfectSlider(Slider):
    tail: Coordinate

    # angular deviation from straight line to end point (-pi, pi)\{0}
    # `deviation` > 0 => arc deviates to the left
    # `deviation` = 0 => technically corresponds to line slider,
    #   but lines are actually parsed as a bezier slider with a single line segment
    deviation: float

    def _cls_str(self) -> str:
        return "P"

    def _slider_str(self) -> str:
        s = ""
        if self.deviation != 0:
            s = f"[deviation={self.deviation:.2f}]"
        return s + f" {self.head} {self.tail}"
    
@dataclass
class LineSegment:
    q: Coordinate

    def __str__(self) -> str:
        return f"Line({self.q})"

@dataclass
class CubicSegment:
    pc: Coordinate
    qc: Coordinate
    q: Coordinate

    def __str__(self) -> str:
        return f"Cubic({self.pc}, {self.qc}, {self.q})"
    
BezierSegment = LineSegment | CubicSegment
    
@dataclass
class BezierSlider(Slider):
    segments: list[BezierSegment]

    def _cls_str(self) -> str:
        return "B"
    
    def _slider_str(self):
        return " " + " ".join(map(str, self.segments))


Timed = Union[
    BeatLen,
    SliderVel,
    Break,
    HitCircle,
    Spinner,
    PerfectSlider,
    BezierSlider,
]