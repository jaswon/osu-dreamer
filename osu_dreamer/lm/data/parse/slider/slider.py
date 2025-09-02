
from typing import Union

import numpy as np

from ...timed import *
from .bezier import parse_bezier

def parse_slider(
    x: int, y: int,
    hit_object_args: tuple[bool, bool, bool, bool],
    raw_curve_spec: str,
    slides: int,
    length: float,
) -> Union[HitCircle, Slider]:
    slider_args = *hit_object_args, -1, slides
    curve_type, *curve_points = raw_curve_spec.split("|")
    ctrl_pts: list[Coordinate] = [(x,y)] + [
        (x,y)
        for p in curve_points
        for x,y,*_ in [ map(lambda s: round(float(s)), p.split(":")) ] 
    ]

    if len(ctrl_pts) == 1:
        # bad slider, return hit circle
        return HitCircle(*hit_object_args,(x,y))
    
    if len(ctrl_pts) == 3 and curve_type == 'P':
        # perfect
        A,B,C = np.array(ctrl_pts)

        ABC = np.cross(B - A, C - B)
        if ABC != 0:

            # recompute tail from length
            a = np.linalg.norm(C - B)
            b = np.linalg.norm(C - A)
            c = np.linalg.norm(B - A)
            s = (a + b + c) / 2.0
            R = a * b * c / 4.0 / np.sqrt(s * (s - a) * (s - b) * (s - c))

            theta = length/R
            if ABC < 0: # clockwise
                theta *= -1

            b1 = a * a * (- a * a + b * b + c * c)
            b2 = b * b * (+ a * a - b * b + c * c)
            b3 = c * c * (+ a * a + b * b - c * c)
            O = np.column_stack((A, B, C)).dot(np.hstack((b1, b2, b3)))
            O /= b1 + b2 + b3

            OA = A - O
            OAP = np.array([-OA[1], OA[0]])
            C = O + OA*np.cos(theta) + OAP*np.sin(theta)

            return PerfectSlider(
                *slider_args, 
                head = tuple(A.round().astype(int).tolist()),
                tail = tuple(C.round().astype(int).tolist()),
                deviation = theta / -2,
            )

        # collinear
        if np.dot(B - A, C - B) > 0:  # A -- B -- C
            # line - remove middle control point
            ctrl_pts.pop(1)
        else:  # A -- C -- B
            # double back slider - repeat middle control point
            ctrl_pts.insert(1, ctrl_pts[1])  # [A,B,B,C]
    
    # bezier
    return parse_bezier(slider_args, ctrl_pts, length)