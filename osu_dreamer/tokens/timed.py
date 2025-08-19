
from dataclasses import dataclass
from typing import Union

Coordinate = tuple[int, int]

@dataclass
class BeatLen:
    """uninherited timing point"""
    ms: float
    meter: int
    
    def __str__(self):
        return f"BeatLen({self.ms}, meter={self.meter})"

@dataclass
class SliderMult:
    """inherited timing point"""
    mult: float
    
    def __str__(self):
        return f"SliderMult({self.mult}x)"

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
        return f" ({self.p[0]},{self.p[1]})"

@dataclass
class _Hold(HitObject):
    duration: int

    def _hold_str(self) -> str:
        return ""

    def _object_str(self) -> str:
        return f"[duration={self.duration}]{self._hold_str()}"

@dataclass
class Spinner(_Hold):
    def _cls_str(self) -> str:
        return "S"

@dataclass
class _Slider(_Hold):
    slides: int

    def _slider_str(self) -> str:
        return ""

    def _hold_str(self):
        return f"[slides={self.slides}]{self._slider_str()}"

@dataclass
class PerfectSlider(_Slider):
    p: Coordinate
    q: Coordinate

    # angular deviation from straight line to end point (-pi, pi)
    # `deviation` > 0 => arc deviates to the left
    # `deviation` = 0 => line slider
    deviation: float

    def _cls_str(self) -> str:
        return "P"

    def _slider_str(self) -> str:
        s = ""
        if self.deviation != 0:
            s = f"[deviation={self.deviation:.2f}]"
        return s + f" ({self.p[0]},{self.p[1]}) ({self.q[0]},{self.q[1]})"
    
@dataclass
class BezierSlider(_Slider):
    shape: list[Coordinate]

    def _cls_str(self) -> str:
        return "B"
    
    def _slider_str(self):
        return " " + " ".join([ f"({x},{y})" for x,y in self.shape ])


Timed = Union[
    BeatLen,
    SliderMult,
    Break,
    HitCircle,
    Spinner,
    PerfectSlider,
    BezierSlider,
]