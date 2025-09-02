
from jaxtyping import Float

from osu_dreamer.osu.bezier import BezierCurve

from ...timed import *
from .fit_poly_cubic import fit_poly_cubic

def parse_bezier(
    slider_args: tuple[bool, bool, bool, bool, int, int], 
    ctrl_pts: list[Coordinate],
):

    assert len(ctrl_pts) >= 2
    try:

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
                    qcq_len = np.linalg.norm(qcq)
                    if qcq_len > .1:
                        u = qcq / qcq_len * delta
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
    

Vec2 = Float[np.ndarray, "2"]

def sample_bezier_slider(ctrl_pts_list: list[Coordinate], length: float):

    ctrl_pts: list[Vec2] = [ np.array(pt) for pt in ctrl_pts_list ]

    # split control points at repeat points
    ctrl_curves: list[list[Vec2]] = []
    last_idx = 0
    for i,p in enumerate(ctrl_pts[1:]):
        if (ctrl_pts[i] == p).all():
            ctrl_curves.append(ctrl_pts[last_idx:i+1])
            last_idx = i+1
    ctrl_curves.append(ctrl_pts[last_idx:])

    path_length = 0
    curves: list[BezierCurve] = []
    for c in ctrl_curves:
        if len(c) < 2:
            # invalid bezier curve spec
            continue

        curve = BezierCurve(np.array(c).T)
        path_length += curve.length
        curves.append(curve)

    # reparametrize based on length
    if length > path_length:
        # longer than defined, extend in a straight line

        # end of bezier curve is tangent to last segment in control points
        last_curve_nodes = curves[-1].p
        p = last_curve_nodes[:, -1]
        v = p - last_curve_nodes[:, -2]

        nodes = np.array([p, p + v / np.linalg.norm(v) * (length - path_length)])
        curves.append(BezierCurve(nodes.T))
        ctrl_pts.extend(nodes)
    else:
        # shorter than defined, shorten last segment
        while path_length - length >= curves[-1].length:
            # slider doesn't reach last segment - shouldn't happen via editor, but possible to encode
            path_length -= curves.pop().length
        curves[-1], _ = curves[-1].split_at( 1 - (path_length - length) / curves[-1].length )

        # recompute control points
        ctrl_pts = [ p for curve in curves for p in curve.p.T ]
        
    cum_t = np.cumsum([ c.length for c in curves ])
    length = cum_t[-1]
    ts = np.linspace(0, length, int(max(4, length//5)))
    idxs = np.searchsorted(cum_t, ts)
    range_start = np.insert(cum_t, 0, 0)[idxs]
    rts = (ts - range_start) / (cum_t[idxs] - range_start)

    return np.stack([
        curves[idx].at(np.array([t]))[:,0]
        for idx, t in zip(idxs, rts)
    ], axis=0), ts / length