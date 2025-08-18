
from typing import Union

import numpy as np

from .timed import HitCircle, PerfectSlider, BezierSlider, Coordinate

def get_perfect_control_point(
    a: Coordinate,
    c: Coordinate,
    deviation: float,
    r: float = .5,
) -> Coordinate:
    """Get a point `B` on the perfect slider"""

    CAB = deviation - np.arcsin(r * np.sin(deviation))

    A = np.array(a)
    AB = np.array(c) - A
    AB_R = np.array([ -AB[1], AB[0] ]) # rotated +pi/2

    AC = r * (np.cos(CAB)*AB + np.sin(CAB)*AB_R)
    return A + AC

def parse_slider(
    x: int, y: int,
    hit_object_args: tuple[bool, bool, bool, bool],
    raw_curve_points: str,
    raw_slides: str,
    raw_length: str,
) -> Union[HitCircle, PerfectSlider, BezierSlider]:
    slider_args = *hit_object_args, int(float(raw_length)), int(raw_slides)
    ctrl_pts: list[Coordinate] = [(x,y)] + [
        (x,y)
        for p in raw_curve_points.split("|")[1:]
        for x,y,*_ in [ map(int, p.split(":")) ] 
    ]
    if len(ctrl_pts) == 1:
        # bad slider, return hit circle
        return HitCircle(*hit_object_args,(x,y))
    elif len(ctrl_pts) == 2:
        # line
        return PerfectSlider(
            *slider_args,
            p = ctrl_pts[0],
            q = ctrl_pts[1],
            deviation = 0,
        )
    elif len(ctrl_pts) == 3:
        # check for perfect
        A,B,C = np.array(ctrl_pts)
        
        BC = C-B
        AB = B-A
        W = BC[0]*AB[1] - BC[1]*AB[0] # wedge product
        D = BC[0]*AB[0] + BC[1]*AB[1] # dot product
        
        if W == 0 and D < 0:  # A -> C -> B
            ctrl_pts.insert(1, ctrl_pts[1]) # [A,B,B,C]
            return BezierSlider(*slider_args, shape=ctrl_pts)
        
        return PerfectSlider(
            *slider_args, 
            p = ctrl_pts[0],
            q = ctrl_pts[2],
            deviation = float(np.arctan2(W,D)),
        )
    else:
        # bezier
        return BezierSlider(*slider_args, shape=ctrl_pts)