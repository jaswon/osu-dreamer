
from typing import Union

import numpy as np

from osu_dreamer.osu.bezier import BezierCurve

from .timed import BezierSegment, HitCircle, PerfectSlider, BezierSlider, Coordinate, LineSegment, CubicSegment
from .fit_poly_cubic import fit_poly_cubic

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
    return tuple(np.round(A + AC).astype(int).tolist())
    
def get_segments(cur_seg: list[Coordinate]) -> list[BezierSegment]:
    if len(cur_seg) == 2:
        # line segment
        p,q = cur_seg
        return [LineSegment(q)]
    elif len(cur_seg) == 3:
        # quadratic -> lift to cubic
        p,c,q = cur_seg
        c1 = round((p[0]+2*c[0])/3), round((p[1]+2*c[1])/3)
        c2 = round((q[0]+2*c[0])/3), round((q[1]+2*c[1])/3)
        return [CubicSegment(c1, c2, q)]
    elif len(cur_seg) == 4:
        # bezier
        _,c1,c2,q = cur_seg
        return [CubicSegment(c1,c2,q)]
    else:
        # higher order - reduce
        return [
            CubicSegment(c1,c2,q)
            for cubic in fit_poly_cubic(BezierCurve(np.array(cur_seg).T), max_allowed_err=10)
            for _,c1,c2,q in [list(map(tuple,cubic.p.T.round().astype(int).tolist()))]
        ]

def parse_bezier(ctrl_pts: list[Coordinate]) -> tuple[Coordinate, list[BezierSegment]]:
    """
    approximates arbitrary poly-beziers as poly-beziers of order 0 (line) or order 3 (cubic)
    """
    assert len(ctrl_pts) >= 2

    segments = []
    head, *ctrl_pts = ctrl_pts
    
    cur_seg = [head]
    for p in ctrl_pts:
        if p == cur_seg[-1]:
            segments.extend(get_segments(cur_seg))
            cur_seg = [p]
        else:
            cur_seg.append(p)
        
    if len(cur_seg) > 1:
        segments.extend(get_segments(cur_seg))

    return head, segments

def parse_slider(
    x: int, y: int,
    hit_object_args: tuple[bool, bool, bool, bool],
    raw_curve_spec: str,
    raw_slides: str,
    raw_length: str,
) -> Union[HitCircle, PerfectSlider, BezierSlider]:
    slider_args = *hit_object_args, int(float(raw_length)), int(raw_slides)
    curve_type, *curve_points = raw_curve_spec.split("|")
    ctrl_pts: list[Coordinate] = [(x,y)] + [
        (x,y)
        for p in curve_points
        for x,y,*_ in [ map(int, p.split(":")) ] 
    ]
    if len(ctrl_pts) == 1:
        # bad slider, return hit circle
        return HitCircle(*hit_object_args,(x,y))
    elif len(ctrl_pts) == 2:
        # line
        return BezierSlider(
            *slider_args,
            head = ctrl_pts[0],
            segments = [ LineSegment(ctrl_pts[1]) ],
        )
    elif len(ctrl_pts) == 3:
        if curve_type == "B":
            return BezierSlider(*slider_args, *parse_bezier(ctrl_pts))

        # check for perfect
        A,B,C = np.array(ctrl_pts)
        
        BC = C-B
        AB = B-A
        W = BC[0]*AB[1] - BC[1]*AB[0] # wedge product
        D = BC[0]*AB[0] + BC[1]*AB[1] # dot product
        
        if W == 0 and D < 0:  # A -> C -> B
            ctrl_pts.insert(1, ctrl_pts[1]) # [A,B,B,C]
            return BezierSlider(*slider_args, *parse_bezier(ctrl_pts))
        
        return PerfectSlider(
            *slider_args, 
            head = ctrl_pts[0],
            tail = ctrl_pts[2],
            deviation = float(np.arctan2(W,D)),
        )
    else:
        # bezier
        return BezierSlider(*slider_args, *parse_bezier(ctrl_pts))