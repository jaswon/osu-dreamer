
from jaxtyping import Float

import numpy as np
from numpy import ndarray

Vec2 = Float[ndarray, "2"]

class Timed:
    def __init__(self, t: int):
        self.t = int(t)

    def __repr__(self):
        return f"{self.t:08}:"
    
    def __lt__(self, other):
        return self.t < other.t


class TimingPoint(Timed):
    def __init__(self, t: int, beat_length: float, slider_mult: float, meter: int):
        super().__init__(t)
        self.beat_length = beat_length
        self.slider_mult = slider_mult
        self.meter = meter
        
    def __repr__(self):
        return " ".join([
            super().__repr__(),
            f"beat_len={self.beat_length}",
            f"slider_mult={self.slider_mult}",
            f"meter={self.meter}",
        ])
    
    def __eq__(self, other: "TimingPoint"):
        return all([
            self.beat_length == other.beat_length,
            self.slider_mult == other.slider_mult,
            self.meter == other.meter,
        ])

class HitObject(Timed):
    def __init__(self, t: int, new_combo: bool):
        super().__init__(t)
        self.new_combo = new_combo

    def __repr__(self):
        return super().__repr__() + (" *" if self.new_combo else "")
    
    def end_time(self) -> int:
        raise NotImplementedError
    
    def start_pos(self) -> Vec2:
        raise NotImplementedError
        
    def end_pos(self) -> Vec2:
        return self.start_pos()


class Circle(HitObject):
    def __init__(self, t: int, new_combo: bool, x: int, y: int):
        super().__init__(t, new_combo)
        self.x = x
        self.y = y

    def __repr__(self):
        return f"{super().__repr__()} Circle({self.x},{self.y})"
    
    def end_time(self) -> int:
        return self.t
    
    def start_pos(self) -> Vec2:
        return np.array([ self.x, self.y ], dtype=float)

class Spinner(HitObject):
    def __init__(self, t: int, new_combo: bool, u: int):
        super().__init__(t, new_combo)
        self.u = u

    def __repr__(self):
        return f"{super().__repr__()} Spinner({self.u})"
    
    def end_time(self) -> int:
        return self.u
    
    def start_pos(self) -> Vec2:
        return np.array([ 256, 192 ], dtype=float)


class Slider(HitObject):
    def __init__(
        self,
        t: int,
        beat_length: float,
        slider_mult: float,
        new_combo: bool,
        slides: int,
        length: float,
    ):
        super().__init__(t, new_combo)
        self.slides = slides
        self.length = length
        self.beat_length = beat_length
        self.slider_mult = slider_mult
        
        # the duration of a single slide (no repeats)
        self.slide_duration = length / (slider_mult * 100) * beat_length
        
    def end_time(self) -> int:
        return int(self.t + self.slide_duration * self.slides)

    def lerp(self, _: Float[ndarray, "L"]) -> Float[ndarray, "L 2"]:
        """
        return cursor pos given fraction of one slide (t=1 means end of slider)
        """
        raise NotImplementedError
    
    def vel(self, _: Float[ndarray, "L"]) -> Float[ndarray, "L 2"]:
        """
        return cursor vel given fraction of one slide (t=1 means end of slider)
        """
        raise NotImplementedError
    
    def start_pos(self) -> Vec2:
        return self.lerp(np.array([0], dtype=float))[0]
    
    def end_pos(self) -> Vec2:
        return self.lerp(np.array([self.slides % 2], dtype=float))[0]
