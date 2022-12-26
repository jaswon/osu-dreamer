from __future__ import annotations
from typing import List, Union

import copy

import numpy as np
import numpy.typing as npt
import bezier

from .hit_objects import Slider, NDIntArray

def approx_eq(a,b):
    return abs(a-b) < 1e-8

def binom_coeffs(n):
    """returns a list of C(n, k) for 0<=k<=n"""
    ret = []
    c = 1.0
    for k in range(n+1):
        ret.append(c)
        c = c*(n+1)/(k+1) - c
    return ret

def from_control_points(
    t: int,
    beat_length: float,
    slider_mult: float,
    new_combo: bool,
    slides: int,
    length: float,
    ctrl_pts: List[NDIntArray],
) -> Slider:
    if len(ctrl_pts) < 2:
        raise Exception(f"bad slider: {ctrl_pts}")
    if len(ctrl_pts) == 2:  # L type
        A, B = ctrl_pts
        return Line(t, beat_length, slider_mult, new_combo, slides, length, A, B)
    if len(ctrl_pts) == 3:  # check P type
        A, B, C = ctrl_pts

        if (B == C).all():
            return Line(t, beat_length, slider_mult, new_combo, slides, length, A, C)

        ABC = np.cross(B - A, C - B)

        if ABC == 0:  # collinear
            if np.dot(B - A, C - B) > 0:  # A -- B -- C
                return Line(t, beat_length, slider_mult, new_combo, slides, length, A, C)
            else:  # A -- C -- B
                ctrl_pts.insert(1, ctrl_pts[1])  # [A,B,B,C]
                return Bezier(t, beat_length, slider_mult, new_combo, slides, length, ctrl_pts)

        a = np.linalg.norm(C - B)
        b = np.linalg.norm(C - A)
        c = np.linalg.norm(B - A)
        s = (a + b + c) / 2.0
        R = a * b * c / 4.0 / np.sqrt(s * (s - a) * (s - b) * (s - c))

        if R > 320 and np.dot(C - B, B - A) < 0:  # circle too large
            return Bezier(t, beat_length, slider_mult, new_combo, slides, length, ctrl_pts)

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

        return Perfect(t, beat_length, slider_mult, new_combo, slides, length, P, R, start_angle, end_angle)
    else:  # B type
        return Bezier(t, beat_length, slider_mult, new_combo, slides, length, ctrl_pts)


class Line(Slider):
    def __init__(
        self,
        t: int,
        beat_length: float,
        slider_mult: float,
        new_combo: bool,
        slides: int,
        length: float,
        start: NDIntArray,
        end: NDIntArray,
    ):
        super().__init__(t, beat_length, slider_mult, new_combo, slides, length)

        self.start = start

        vec = end - self.start
        self.end = self.start + vec / np.linalg.norm(vec) * length
        # self.end = self.end.round(0).astype(np.integer)

    def __repr__(self):
        return f"{super().__repr__()} Line[*{self.slides}]({self.start} -> {self.end})"

    def lerp(self, t: float) -> NDIntArray:
        ret = (1 - t) * self.start + t * self.end
        return ret.round(0).astype(np.integer)

    def vel(self, t: float) -> NDIntArray:
        return (self.end - self.start) / (self.slide_duration / self.slides)


class Perfect(Slider):
    def __init__(
        self,
        t: int,
        beat_length: float,
        slider_mult: float,
        new_combo: bool,
        slides: int,
        length: float,
        center: NDIntArray,
        radius: float,
        start: float,
        end: float,
    ):
        super().__init__(t, beat_length, slider_mult, new_combo, slides, length)
        self.center = center
        self.radius = radius
        self.start = start

        self.end = start + length / radius * np.sign(end - start)

    def __repr__(self):
        return f"{super().__repr__()} Perfect[*{self.slides}](O:{self.center} R:{self.radius} {self.start} -> {self.end})"

    def lerp(self, t: float) -> NDIntArray:
        angle = (1 - t) * self.start + t * self.end
        ret = self.center + self.radius * np.array([np.cos(angle), np.sin(angle)])
        return ret.round(0).astype(np.integer)

    def vel(self, t: float) ->  NDIntArray:
        angle = (1 - t) * self.start + t * self.end
        return self.radius * np.array([-np.sin(angle), np.cos(angle)]) / (self.slide_duration / self.slides)



class Bezier(Slider):
    SEG_LEN = 10

    def __repr__(self):
        return f"{super().__repr__()} Bezier[*{self.slides}]({self.ctrl_pts})"

    def __init__(
        self,
        t: int,
        beat_length: float,
        slider_mult: float,
        new_combo: bool,
        slides: int,
        length: float,
        ctrl_pts, # n x 2
    ):
        super().__init__(t, beat_length, slider_mult, new_combo, slides, length)

        self.ctrl_pts = ctrl_pts

        # split control points at repeat points
        ctrl_curves: List[List[NDIntArray]] = []
        last_idx = 0
        for i,p in enumerate(ctrl_pts[1:]):
            if (ctrl_pts[i] == p).all():
                ctrl_curves.append(ctrl_pts[last_idx:i+1])
                last_idx = i+1
        ctrl_curves.append(ctrl_pts[last_idx:])

        total_len = 0
        curves = []
        for c in ctrl_curves:
            if len(c) < 2:
                # invalid bezier curve spec
                continue

            nodes = np.array(c).T # 2 x n
            curve = bezier.Curve.from_nodes(nodes)
            total_len += curve.length
            curves.append(curve)

        tail_len = self.length - total_len
        if tail_len > 0:
            # computed length less than defined length
            # -> extend slider in a straight line

            # end of bezier curve is tangent to last segment in control points
            last_curve_nodes = curves[-1].nodes
            p = last_curve_nodes[:, -1]
            v = p - last_curve_nodes[:, -2]

            nodes = np.array([p, p + v / np.linalg.norm(v) * tail_len]).T

            curve = bezier.Curve.from_nodes(nodes)

            assert approx_eq(curve.length, tail_len), f"{curve.length} != {tail_len}"
            curves.append(curve)
            
        self.path_segments = curves
        self.cum_t = np.cumsum([ c.length for c in curves ]) / self.length
        self.cum_t[-1] = 1.

    def curve_reparameterize(self, t: float) -> "(int, float)":
        """
        converts the parameter to an index into the sequence of curves and a new parameter localized to that curve
        """
        idx = np.searchsorted(self.cum_t, min(1, max(0, t)))
        assert idx < len(self.cum_t), (idx, t, self.cum_t)

        range_start = np.insert(self.cum_t, 0, 0)[idx]
        range_end = self.cum_t[idx]

        t = (t - range_start) / (range_end - range_start)
        return idx, t

    def lerp(self, t: float):
        idx, t = self.curve_reparameterize(t)
        return self.path_segments[idx].evaluate(t)[:,0].round(0).astype(np.integer)

    def vel(self, t: float) -> NDIntArray:
        idx, t = self.curve_reparameterize(t)
        return self.path_segments[idx].evaluate_hodograph(t)[:,0].round(0).astype(np.integer) / (self.slide_duration / self.slides)