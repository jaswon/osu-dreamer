
from osu_dreamer.osu.bezier import BezierCurve

from ...timed import *
from .reduce import reduce_to_poly_cubic
from .fit import sample_bezier_slider, fit_to_poly_cubic

def parse_bezier(
    slider_args: tuple[bool, bool, bool, bool, int, int], 
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
        poly_cubic = fit_to_poly_cubic(ps, ts)
        return BezierSlider(*slider_args, head=head, segments=[
            CubicSegment(q,c1,c2)
            for cubic in poly_cubic
            for _,c1,c2,q in [list(map(tuple,cubic.p.T.round().astype(int).tolist()))]
        ])
    
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
        heads.append(seg.q)
    heads.pop()

    if abs(length - path_length) < 15:
        # negligible difference
        pass
    elif length > path_length:
        delta = length - path_length

        if isinstance(segments[-1], LineSegment):
            # extend line segment
            p = np.array(heads[-1])
            pq = np.array(segments[-1].q) - p
            pq = pq / seg_lens[-1] * (seg_lens[-1] + delta)
            q = p + pq
            segments[-1].q = tuple(q.round().astype(int).tolist())
        else:
            # append line segment
            match segments[-1]:
                case QuadraticSegment(q,c): seg_curve = BezierCurve(np.array([heads[-1], c, q]).T)
                case CubicSegment(q,c1,c2): seg_curve = BezierCurve(np.array([heads[-1], c1, c2, q]).T)
            d = seg_curve.hodo().at(np.array([1.]))[:,0]
            d_len = np.linalg.norm(d)
            q = np.array(q) + d / d_len * delta
            segments.append(LineSegment(tuple(q.round().astype(int).tolist())))

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
            match segments[-1]:
                case LineSegment(q):
                    # shorten line segment
                    pq = np.array(q) - p
                    pq = pq / seg_len * (seg_len - delta)
                    q = p + pq
                    segments[-1].q = tuple(q.round().astype(int).tolist())
                case QuadraticSegment(q,c):
                    # split quadratic segment
                    cubic = BezierCurve(np.array([p,c,q]).T)
                    split, _ = cubic.split_at_length(1-delta/seg_len)
                    _,c,q = split.p.T
                    segments[-1].q = tuple(q.round().astype(int).tolist())
                    segments[-1].c = tuple(c.round().astype(int).tolist())
                case CubicSegment(q,c1,c2):
                    # split cubic segment
                    cubic = BezierCurve(np.array([p,c1,c2,q]).T)
                    split, _ = cubic.split_at_length(1-delta/seg_len)
                    _,pc,qc,q = split.p.T
                    segments[-1].q = tuple(q.round().astype(int).tolist())
                    segments[-1].pc = tuple(pc.round().astype(int).tolist())
                    segments[-1].qc = tuple(qc.round().astype(int).tolist())

    # check for lines/polylines
    if all(isinstance(seg, LineSegment) for seg in segments):
        return PolyLineSlider(*slider_args, head=head, vertices=[ seg.q for seg in segments ])

    return BezierSlider(*slider_args, head, segments)
    
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
        
        return [QuadraticSegment(q,c)]
    elif len(cur_seg) == 4:
        # bezier
        _,c1,c2,q = cur_seg
        return [CubicSegment(q,c1,c2)]
    else:
        # higher order - reduce
        try:
            poly_cubic = reduce_to_poly_cubic(BezierCurve(np.array(cur_seg).T))
        except Exception as e:
            raise Exception(cur_seg) from e
        
        return [
            CubicSegment(q,c1,c2)
            for cubic in poly_cubic
            for _,c1,c2,q in [list(map(tuple,cubic.p.T.round().astype(int).tolist()))]
        ]
    