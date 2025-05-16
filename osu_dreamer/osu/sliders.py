
from jaxtyping import Float, Int

import numpy as np
from numpy import ndarray

from .hit_objects import Slider, Vec2
from .bezier import BezierCurve

def from_control_points(
    t: int,
    beat_length: float,
    slider_mult: float,
    new_combo: bool,
    hit_sound: int,
    slides: int,
    length: float,
    ctrl_pts: list[Vec2],
) -> Slider:
    slider_args = t, beat_length, slider_mult, new_combo, hit_sound, slides, length, ctrl_pts

    if len(ctrl_pts) < 2:
        raise Exception(f"bad slider: {ctrl_pts}")
    if len(ctrl_pts) == 2:  # L type
        A, B = ctrl_pts
        return Line(*slider_args, A, B)
    if len(ctrl_pts) == 3:  # check P type
        A, B, C = ctrl_pts

        if (B == C).all():
            ctrl_pts.pop(1)
            return Line(*slider_args, A, C)

        ABC = np.cross(B - A, C - B)

        if ABC == 0:  # collinear
            if np.dot(B - A, C - B) > 0:  # A -- B -- C
                ctrl_pts.pop(1)
                return Line(*slider_args, A, C)
            else:  # A -- C -- B
                ctrl_pts.insert(1, ctrl_pts[1])  # [A,B,B,C]
                return Bezier(*slider_args)

        a = np.linalg.norm(C - B)
        b = np.linalg.norm(C - A)
        c = np.linalg.norm(B - A)
        s = (a + b + c) / 2.0
        R = a * b * c / 4.0 / np.sqrt(s * (s - a) * (s - b) * (s - c))

        if R > 320 and np.dot(C - B, B - A) < 0:  # circle too large
            return Bezier(*slider_args)

        b1 = a * a * (b * b + c * c - a * a)
        b2 = b * b * (a * a + c * c - b * b)
        b3 = c * c * (a * a + b * b - c * c)
        P = np.column_stack((A, B, C)).dot(np.hstack((b1, b2, b3)))
        P /= b1 + b2 + b3

        start_angle = np.arctan2(*(A - P)[[1, 0]])
        end_angle = np.arctan2(*(C - P)[[1, 0]])

        if ABC < 0:  # clockwise
            while end_angle > start_angle:
                end_angle -= 2 * np.pi
        else:  # counter-clockwise
            while start_angle > end_angle:
                start_angle -= 2 * np.pi

        return Perfect(*slider_args, P, R, start_angle, end_angle)
    else:  # B type
        return Bezier(*slider_args)

class Line(Slider):
    def __init__(
        self,
        t: int,
        beat_length: float,
        slider_mult: float,
        new_combo: bool,
        hit_sound: int,
        slides: int,
        length: float,
        ctrl_pts: list[Vec2],
        start: Vec2,
        end: Vec2,
    ):
        super().__init__(t, beat_length, slider_mult, new_combo, hit_sound, slides, length, ctrl_pts)
        self.start = start

        if length > 0:
            # reparametrize based on length
            vec = end - self.start
            self.end = self.ctrl_pts[-1] = self.start + vec / np.linalg.norm(vec) * length
        else: 
            # compute length based on parameters
            self.end = end
            self.length = np.linalg.norm(self.end - self.start)

    def __repr__(self):
        return f"{super().__repr__()} Line[*{self.slides}]({self.start} -> {self.end})"

    def lerp(self, t: Float[ndarray, "L"]) -> Float[ndarray, "L 2"]:
        return (1 - t[:,None]) * self.start + t[:,None] * self.end

    def vel(self, t: Float[ndarray, "L"]) -> Float[ndarray, "L 2"]:
        vel = (self.end - self.start) / self.slide_duration
        return vel[None].repeat(t.shape[0], axis=0)


class Perfect(Slider):
    def __init__(
        self,
        t: int,
        beat_length: float,
        slider_mult: float,
        new_combo: bool,
        hit_sound: int, 
        slides: int,
        length: float,
        ctrl_pts: list[Vec2],
        center: Vec2,
        radius: float,
        start: float,
        end: float,
    ):
        super().__init__(t, beat_length, slider_mult, new_combo, hit_sound, slides, length, ctrl_pts)
        self.center = center
        self.radius = radius
        self.start = start

        if length > 0:
            # reparametrize based on length
            self.end = start + length / radius * np.sign(end - start)
            self.ctrl_pts[-1] = self.lerp(np.array([1.]))[0]
        else: 
            # compute length based on parameters
            self.length = abs(end - start) * radius

    def __repr__(self):
        return f"{super().__repr__()} Perfect[*{self.slides}](O:{self.center} R:{self.radius} {self.start} -> {self.end})"

    def lerp(self, t: Float[ndarray, "L"]) -> Float[ndarray, "L 2"]:
        angle = (1 - t) * self.start + t * self.end
        return self.center + self.radius * np.stack([np.cos(angle), np.sin(angle)], axis=1)

    def vel(self, t: Float[ndarray, "L"]) -> Float[ndarray, "L 2"]:
        angle = (1 - t) * self.start + t * self.end
        return self.radius * np.stack([-np.sin(angle), np.cos(angle)], axis=1) * (self.end - self.start) / self.slide_duration


class Bezier(Slider):
    def __init__(
        self,
        t: int,
        beat_length: float,
        slider_mult: float,
        new_combo: bool,
        hit_sound: int,
        slides: int,
        length: float,
        ctrl_pts: list[Vec2], # n x 2
    ):
        super().__init__(t, beat_length, slider_mult, new_combo, hit_sound, slides, length, ctrl_pts)

        # split control points at repeat points
        ctrl_curves: list[list[Vec2]] = []
        last_idx = 0
        for i,p in enumerate(ctrl_pts[1:]):
            if (ctrl_pts[i] == p).all():
                ctrl_curves.append(ctrl_pts[last_idx:i+1])
                last_idx = i+1
        ctrl_curves.append(ctrl_pts[last_idx:])

        total_len = 0
        curves: list[BezierCurve] = []
        for c in ctrl_curves:
            if len(c) < 2:
                # invalid bezier curve spec
                continue

            curve = BezierCurve(np.array(c).T)
            total_len += curve.length
            curves.append(curve)

        if length > 0:
            # reparametrize based on length
            if abs(length - total_len) < 10:
                # close enough, ignore
                pass
            elif length > total_len:
                # longer than defined, extend in a straight line

                # end of bezier curve is tangent to last segment in control points
                last_curve_nodes = curves[-1].p
                p = last_curve_nodes[:, -1]
                v = p - last_curve_nodes[:, -2]

                nodes = np.array([p, p + v / np.linalg.norm(v) * (length - total_len)])
                curves.append(BezierCurve(nodes.T))
                self.ctrl_pts.extend(nodes)
            else:
                # shorter than defined, shorten last segment
                while total_len - length >= curves[-1].length:
                    # slider doesn't reach last segment - shouldn't happen via editor, but possible to encode
                    total_len -= curves.pop().length
                curves[-1], _ = curves[-1].split_at( 1 - (total_len - length) / curves[-1].length )

                # recompute control points
                self.ctrl_pts = [ p for curve in curves for p in curve.p.T ]
        else: 
            # compute length based on parameters
            self.length = total_len
            
        self.path_segments: list[BezierCurve] = curves
        self.cum_t = np.cumsum([ c.length for c in curves ])
        self.cum_t /= self.cum_t[-1]

    def __repr__(self):
        return f"{super().__repr__()} Bezier[*{self.slides}]({self.ctrl_pts})"

    def curve_reparameterize(self, t: Float[ndarray, "L"]) -> tuple[Int[ndarray, "L"], Float[ndarray, "L"]]:
        """
        converts the parameter to a segment index and a new parameter localized to that segment
        """
        t = np.clip(t, 0, 1)
        idx = np.searchsorted(self.cum_t, t)

        range_start = np.insert(self.cum_t, 0, 0)[idx]
        range_end = self.cum_t[idx]

        t = (t - range_start) / (range_end - range_start)
        return idx, t

    def lerp(self, t: Float[ndarray, "L"]) -> Float[ndarray, "L 2"]:
        return np.stack([
            self.path_segments[idx].at(np.array([t]))[:,0]
            for idx, t in zip(*self.curve_reparameterize(t))
        ], axis=0)

    def vel(self, t: Float[ndarray, "L"]) -> Float[ndarray, "L 2"]:
        return np.stack([
            self.path_segments[idx].hodo().at(np.array([t]))[:,0] / self.slide_duration
            for idx, t in zip(*self.curve_reparameterize(t))
        ], axis=0)