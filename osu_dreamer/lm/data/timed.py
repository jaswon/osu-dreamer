
from dataclasses import dataclass
from typing import Union

import numpy as np

from osu_dreamer.osu.bezier import BezierCurve

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

    def length(self) -> float:
        raise NotImplementedError

    def vel(self) -> float:
        return self.length() * self.slides / self.duration

    def _slider_str(self) -> str:
        return ""

    def _hold_str(self):
        return f"[slides={self.slides}]{self._slider_str()}"

@dataclass
class PerfectSlider(Slider):
    tail: Coordinate

    # direction of the tangent at the start of the slider as the angle above the chord from start to end
    # `deviation` > 0 => arc deviates to the left
    # `deviation` = pi/2 => arc is a semi-circle
    # `deviation` = 0 => technically corresponds to line slider,
    #   but lines are actually parsed as a bezier slider with a single line segment
    deviation: float # (-pi, pi)\{0}

    def _cls_str(self) -> str:
        return "P"

    def _slider_str(self) -> str:
        s = ""
        if self.deviation != 0:
            s = f"[deviation={self.deviation:.2f}]"
        return s + f" {self.head} {self.tail}"
    
    def get_control_point(self) -> Coordinate:
        """Get a point `B` on the perfect slider"""

        A = np.array(self.head)
        C = np.array(self.tail)
        AC = C - A

        r = np.sin(self.deviation/2) / np.sin(self.deviation)
        r = min(r, 100/np.linalg.norm(AC))

        CAB = self.deviation - np.arcsin(r * np.sin(self.deviation))

        AC_R = np.array([ -AC[1], AC[0] ]) # rotated +pi/2

        AC = r * (np.cos(CAB)*AC + np.sin(CAB)*AC_R)
        return tuple(np.round(A + AC).astype(int).tolist())
    
    def length(self) -> float:
        a = np.array(self.head)
        c = np.array(self.tail)
        ac_dist = np.linalg.norm(c - a).item()

        if abs(self.deviation) < 1e-8:
            return ac_dist
        
        # ref: alternate segment theorem
        return ac_dist * self.deviation / np.sin(self.deviation)
    
@dataclass
class LineSegment:
    q: Coordinate

    def __str__(self) -> str:
        return f"Line({self.q})"
    
    def length(self, p: Coordinate) -> float:
        return np.linalg.norm(np.array(p) - np.array(self.q)).item()

@dataclass
class CubicSegment:
    pc: Coordinate
    qc: Coordinate
    q: Coordinate

    def __str__(self) -> str:
        return f"Cubic({self.pc}, {self.qc}, {self.q})"
    
    def length(self, p: Coordinate) -> float:
        return BezierCurve(np.array([p,self.pc,self.qc,self.q]).T).length
    
BezierSegment = LineSegment | CubicSegment
    
@dataclass
class BezierSlider(Slider):
    segments: list[BezierSegment]

    def _cls_str(self) -> str:
        return "B"
    
    def _slider_str(self):
        return f" {self.head} {" ".join(map(str, self.segments))}"
    
    def length(self) -> float:
        length = 0
        p = self.head
        for seg in self.segments:
            length += seg.length(p)
            p = seg.q
        return length

Timed = Union[
    BeatLen,
    SliderVel,
    Break,
    HitCircle,
    Spinner,
    PerfectSlider,
    BezierSlider,
]