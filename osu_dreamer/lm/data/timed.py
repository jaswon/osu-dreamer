
from dataclasses import KW_ONLY, dataclass

import numpy as np

from osu_dreamer.osu.bezier import BezierCurve

Coordinate = tuple[int, int]

@dataclass
class Timed:
    t: int
    
@dataclass
class Hold(Timed):
    u: int

    @property
    def duration(self) -> int:
        return max(0, self.u-self.t)

@dataclass
class BeatLen(Timed):
    """uninherited timing point"""
    ms: float

@dataclass
class SliderVel(Timed):
    """inherited timing point"""
    vel: float

@dataclass
class Break(Hold):
    pass

@dataclass
class HitObject(Timed):
    new_combo: bool
    whistle: bool
    finish: bool
    clap: bool

@dataclass
class HitCircle(HitObject):
    p: Coordinate

@dataclass
class Spinner(HitObject, Hold):
    pass

@dataclass
class Slider(HitObject, Hold):
    slides: int
    head: Coordinate

    def length(self) -> float:
        raise NotImplementedError

    def vel(self) -> float:
        return self.length() * self.slides / self.duration

@dataclass
class BezierSegment:
    ctrl: list[Coordinate]

    def length(self, p: Coordinate) -> float:
        return BezierCurve(np.array([p, *self.ctrl]).astype(float).T).length 
    
@dataclass
class BezierSlider(Slider):
    segments: list[BezierSegment]
    
    def length(self) -> float:
        length = 0
        p = self.head
        for seg in self.segments:
            length += seg.length(p)
            p = seg.ctrl[-1]
        return length
    
@dataclass
class PolyLineSlider(Slider):
    vertices: list[Coordinate]
    
    def length(self) -> float:
        length = 0
        start = self.head
        for v in self.vertices:
            length += np.linalg.norm(np.array(start) - np.array(v)).item()
            start = v
        return length
    
@dataclass
class PerfectSlider(Slider):
    tail: Coordinate

    # direction of the tangent at the start of the slider as the angle above the chord from start to end
    # `deviation` > 0 => arc deviates to the left
    # `deviation` = pi/2 => arc is a semi-circle
    # `deviation` = 0 => technically corresponds to line slider,
    #   but lines are actually parsed as a bezier slider with a single line segment
    deviation: float # (-pi, pi)\{0}
    
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
    