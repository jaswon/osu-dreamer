
from typing import Union

import numpy as np

from osu_dreamer.osu.bezier import BezierCurve

from ..timed import *
from .fit_poly_cubic import fit_poly_cubic

def parse_slider(
    x: int, y: int,
    hit_object_args: tuple[bool, bool, bool, bool],
    raw_curve_spec: str,
    slides: int,
    length: float,
) -> Union[HitCircle, PerfectSlider, BezierSlider]:
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
    try:
        head, segments = parse_bezier(ctrl_pts)
    except Exception as e:
        raise Exception(ctrl_pts) from e

    # HACK: ignore many-segment beziers
    if len(segments) > 16:
        segments: list[BezierSegment] = [LineSegment(ctrl_pts[-1])]
        length = segments[0].length(head)

    # recompute tail from length
    path_length = 0
    seg_lens = []
    heads = [head]
    for seg in segments:
        seg_len = seg.length(heads[-1])
        seg_lens.append(seg_len)
        path_length += seg_len
        heads.append(seg.q)
    heads.pop()

    if length > path_length:
        delta = length - path_length
        match segments[-1]:
            case LineSegment(q):
                # extend line segment
                p = np.array(heads[-1])
                pq = np.array(q) - p
                pq = pq / seg_len * (seg_len + delta)
                q = p + pq
                segments[-1] = LineSegment(tuple(q.round().astype(int).tolist()))
            case CubicSegment(_,qc,q):
                # append line segment
                if delta > 10:
                    qcq = np.array(q) - np.array(qc)
                    u = qcq / np.linalg.norm(qcq) * delta
                    q = np.array(q) + u
                    segments.append(LineSegment(tuple(q.round().astype(int).tolist())))

    elif length < path_length:
        while path_length - length >= seg_lens[-1]:
            # slider doesn't reach last segment - shouldn't happen via editor, but possible to encode
            path_length -= seg_lens.pop()
            segments.pop()
            heads.pop()

        delta = path_length - length
        seg_len = seg_lens[-1]
        p = np.array(heads[-1])
        match segments[-1]:
            case LineSegment(q):
                # shorten line segment
                pq = np.array(q) - p
                pq = pq / seg_len * (seg_len - delta)
                q = p + pq
                segments[-1] = LineSegment(tuple(q.round().astype(int).tolist()))
            case CubicSegment(pc,qc,q):
                # split cubic segment
                cubic = BezierCurve(np.array([p,pc,qc,q]).T)
                split, _ = cubic.split_at(1-delta/seg_len)
                _,pc,qc,q = split.p.T
                segments[-1] = CubicSegment(
                    pc = tuple(pc.round().astype(int).tolist()),
                    qc = tuple(qc.round().astype(int).tolist()),
                    q = tuple(q.round().astype(int).tolist()),
                )

    return BezierSlider(*slider_args, head, segments)

def parse_bezier(ctrl_pts: list[Coordinate]) -> tuple[Coordinate, list[BezierSegment]]:
    """
    approximates arbitrary poly-beziers as poly-beziers of order 1 (line) or order 3 (cubic)
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
    
def get_segments(cur_seg: list[Coordinate]) -> list[BezierSegment]:
    if len(cur_seg) < 2:
        # invalid
        return []
    if len(cur_seg) == 2:
        # line segment
        p,q = cur_seg
        return [LineSegment(q)]
    elif len(cur_seg) == 3:
        # quadratic
        p,c,q = cur_seg

        # check for degenerate
        if np.linalg.norm(np.array(q) - np.array(p)) < .1:
            # same endpoint - split into two line segments
            u = p + .5 * (np.array(c) - np.array(p))
            return [
                LineSegment(tuple(u.round().astype(int).tolist())),
                LineSegment(q),
            ]

        # lift to cubic
        c1 = round((p[0]+2*c[0])/3), round((p[1]+2*c[1])/3)
        c2 = round((q[0]+2*c[0])/3), round((q[1]+2*c[1])/3)
        return [CubicSegment(c1, c2, q)]
    elif len(cur_seg) == 4:
        # bezier
        _,c1,c2,q = cur_seg
        return [CubicSegment(c1,c2,q)]
    else:
        # higher order - reduce
        try:
            poly_cubic = fit_poly_cubic(BezierCurve(np.array(cur_seg).T), max_allowed_err=10)
        except Exception as e:
            raise Exception(cur_seg) from e
        
        return [
            CubicSegment(c1,c2,q)
            for cubic in poly_cubic
            for _,c1,c2,q in [list(map(tuple,cubic.p.T.round().astype(int).tolist()))]
        ]