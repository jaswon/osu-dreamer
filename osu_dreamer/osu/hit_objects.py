from __future__ import annotations
from typing import Tuple

import numpy as np
import numpy.typing as npt

NDIntArray = npt.NDArray[np.integer]

class Timed:
    def __init__(self, t: int):
        self.t = t

    def __repr__(self):
        return f"{self.t:08}:"


class TimingPoint(Timed):
    def __init__(self, t: int, x: float):
        super().__init__(t)
        self.x = x


class Uninherited(TimingPoint):
    def __init__(self, t: int, x: float, meter: int):
        """
        x: duration of a beat in ms
        meter: beats in a measure
        """
        super().__init__(t, x)
        self.meter = meter

    def __repr__(self):
        return f"{super().__repr__()} {self.x} (x{self.meter})"


class Inherited(TimingPoint):
    def __init__(self, t: int, x: float):
        """
        x: slider speed multiplier (eg 2 = twice as fast)
        """
        super().__init__(t, x)

    def __repr__(self):
        return f"{super().__repr__()} *{self.x}"

class HitObject(Timed):
    pass


class Circle(HitObject):
    def __init__(self, t: int, x: int, y: int):
        super().__init__(t)
        self.x = x
        self.y = y

    def __repr__(self):
        return f"{super().__repr__()} Circle({self.x},{self.y})"


class Spinner(HitObject):
    def __init__(self, t: int, u: int):
        super().__init__(t)
        self.u = u

    def __repr__(self):
        return f"{super().__repr__()} Spinner({self.u})"


class Slider(HitObject):
    def __init__(self, t: int, slides: int, length: float):
        super().__init__(t)
        self.slides = slides
        self.length = length

    def lerp(self, _: float) -> NDIntArray:
        """
        return cursor pos given fraction of one slide (t=1 means end of slider)
        """
        raise NotImplementedError
