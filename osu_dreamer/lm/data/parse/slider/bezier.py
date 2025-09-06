
from typing import Sequence
from osu_dreamer.osu.bezier import BezierCurve

from ...timed import *
from .reduce import reduce_to_poly_cubic
from .fit import sample_bezier_slider, fit_to_poly_cubic

def to_segments(curves: Sequence[BezierCurve]) -> list[BezierSegment]:
    segments = []
    for curve in curves:
        for curve in ( # check for degenerate
            curve.split_at_length(.5) 
            if np.linalg.norm(curve.p[:,0] - curve.p[:,-1]) < .1 
            else [curve]
        ):
            ctrls = list(map(tuple,curve.p.T.round().astype(int).tolist()))
            segments.append(BezierSegment(ctrls[1:]))
    return segments


def parse_bezier(
    slider_args: tuple[int, int, bool, bool, bool, bool, int], 
    ctrl_pts: list[Coordinate],
    length: float,
):
    assert len(ctrl_pts) >= 2

    # split bezier into segments
    ctrl_curves: list[list[Coordinate]] = []
    head, *ctrl_pts = ctrl_pts
    
    curve = [head]
    for p in ctrl_pts:
        if p == curve[-1]:
            ctrl_curves.append(curve)
            curve = [p]
        else:
            curve.append(p)
        
    if len(curve) > 1:
        ctrl_curves.append(curve)

    # check for many segment sliders
    if len(ctrl_curves) > 6:
        # fit poly-cubic to interpolated points
        ps, ts = sample_bezier_slider(ctrl_curves, length)
        return BezierSlider(*slider_args, head=head, segments=to_segments(fit_to_poly_cubic(ps, ts)))
    
    # parse and reduce to poly-cubic
    segments = [ seg for curve in ctrl_curves for seg in get_segments(curve) ]

    # recompute tail from length
    path_length = 0
    seg_lens = []
    heads = [head]
    for seg in segments:
        seg_len = seg.length(heads[-1])
        seg_lens.append(seg_len)
        path_length += seg_len
        heads.append(seg.ctrl[-1])
    heads.pop()

    if abs(length - path_length) < 15:
        # negligible difference
        pass
    elif length > path_length:
        delta = length - path_length

        if len(seg.ctrl) == 0:
            # extend line segment
            p = np.array(heads[-1])
            pq = np.array(seg.ctrl[-1]) - p
            pq = pq / seg_lens[-1] * (seg_lens[-1] + delta)
            q = p + pq
            seg.ctrl[-1] = tuple(q.round().astype(int).tolist())
        else:
            # append line segment
            seg_curve = BezierCurve(np.array([heads[-1], *seg.ctrl]).T)
            d = seg_curve.hodo().at(np.array([1.]))[:,0]
            d_len = np.linalg.norm(d)
            q = np.array(seg.ctrl[-1]) + d / d_len * delta
            segments.append(BezierSegment([tuple(q.round().astype(int).tolist())]))

    elif length < path_length:
        while path_length - length >= seg_lens[-1]:
            # slider doesn't reach last segment - shouldn't happen via editor, but possible to encode
            path_length -= seg_lens.pop()
            segments.pop()
            heads.pop()

        delta = path_length - length
        seg_len = seg_lens[-1]

        if seg_len - delta < 10:
            # shortening last segment results in a small segment, just remove
            segments.pop()
        else:
            p = np.array(heads[-1])
            curve = BezierCurve(np.array([p,*segments[-1].ctrl]).T)
            split, _ = curve.split_at_length(1-delta/seg_len)
            segments[-1].ctrl = [
                tuple(c.round().astype(int).tolist())
                for c in split.p.T[1:]
            ] 

    # check for lines/polylines
    if all(len(seg.ctrl) == 1 for seg in segments):
        return PolyLineSlider(*slider_args, head=head, vertices=[ seg.ctrl[0] for seg in segments ])

    return BezierSlider(*slider_args, head, segments)
    
def get_segments(cur_seg: list[Coordinate]) -> list[BezierSegment]:
    if len(cur_seg) < 2:
        # invalid
        return []
    
    if len(cur_seg) > 4:
        # higher order - reduce
        try:
            poly_cubic = reduce_to_poly_cubic(BezierCurve(np.array(cur_seg).T))
        except Exception as e:
            raise Exception(cur_seg) from e
        
        return to_segments(poly_cubic)
    
    # check for degenerate
    if np.linalg.norm(np.array(cur_seg[-1]) - np.array(cur_seg[0])) < .1:
        # quadratic splits into two line segments
        if len(cur_seg) == 3:
            p,c,q = cur_seg
            u = p + .5 * (np.array(c) - np.array(p))
            u = tuple(u.round().astype(int).tolist())
            return [BezierSegment([u]), BezierSegment([q])]
        
        # cubic splits into two cubic segments
        curve = BezierCurve(np.array(cur_seg).T)
        return to_segments(curve.split_at_length(.5))

    return [BezierSegment(cur_seg[1:])]