
from jaxtyping import Float

import numpy as np

from osu_dreamer.osu.bezier import BezierCurve

from ...timed import *

Vec2 = Float[np.ndarray, "2"]


def fit_to_poly_cubic(
    points: Float[np.ndarray, "T 2"],
    ts: Float[np.ndarray, "T"],
    max_allowed_err: float = 10., 
) -> list[BezierCurve]:
    """fits a cubic poly-Bezier curve to points"""

    cubic = fit_to_cubic(points, ts)

    if cubic.p.shape[1] < 4:
        return [cubic]

    # find max error
    errs = np.linalg.norm(cubic.at(ts).T - points, axis=1) # L
    max_err_i = errs.argmax()

    if errs[max_err_i] < max_allowed_err:
        return [cubic]
    
    # split at max error location and fit recursively
    points_a = points[:max_err_i+1]
    ts_a = ts[:max_err_i+1] / ts[max_err_i]

    points_b = points[max_err_i:]
    ts_b = (ts[max_err_i:] - ts[max_err_i]) / (1 - ts[max_err_i])

    return [
        *fit_to_poly_cubic(points_a, ts_a, max_allowed_err),
        *fit_to_poly_cubic(points_b, ts_b, max_allowed_err),
    ]

def fit_to_cubic(
    points: Float[np.ndarray, "T 2"],
    ts: Float[np.ndarray, "T"],
) -> BezierCurve:
    """
    Fits a cubic Bezier curve to a set of 2D points with corresponding parameter values.

    The method uses linear least squares to find the optimal inner control points (P1 and P2)
    that minimize the distance between the curve and the provided points, given that the
    curve must pass through the first and last points (P0 and P3).

    Args:
        points: An array of T 2D points to fit the curve to.
        ts: An array of T parameter values (typically in [0, 1]) corresponding to each point.

    Returns:
        A BezierCurve object representing the best-fit cubic Bezier curve.
    """

    assert len(points) >= 2, points

    # The first and last control points are the start and end of the point sequence.
    p0 = points[0]
    p3 = points[-1]

    # If there are only two points, the curve is a straight line.
    if len(points) == 2:
        return BezierCurve(np.stack([p0, p3], axis=1))

    # If there are three points, fit a quadratic and elevate degree.
    if len(points) == 3:
        d1 = points[1]
        t1 = ts[1]

        # check for collinearity
        v1 = d1 - p0
        v2 = p3 - p0
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        # if points are collinear or coincident, return a straight line
        # this avoids division by zero and degenerate cubics
        if v1_norm < 1e-6 or v2_norm < 1e-6 or \
           (v1_norm * v2_norm > 1e-9 and np.abs(v1[0] * v2[1] - v1[1] * v2[0]) / (v1_norm * v2_norm) < 1e-6):
            return BezierCurve(np.stack([p0, p3], axis=1))

        # Solve for the middle control point of a quadratic Bezier
        # Q(t) = (1-t)^2*p0 + 2t(1-t)*q1 + t^2*p3
        # such that Q(t1) = d1
        c_num = d1 - (1 - t1)**2 * p0 - t1**2 * p3
        c_den = 2 * t1 * (1 - t1)
        
        c = c_num / c_den
        
        return BezierCurve(np.stack([p0, c, p3], axis=1))

    # For more than three points, we set up a linear least squares problem.
    # A cubic Bezier is C(t) = B0(t)P0 + B1(t)P1 + B2(t)P2 + B3(t)P3,
    # where Bi are the Bernstein basis polynomials.
    # We know P0 and P3. We solve for P1 and P2.
    # For each point D_i at t_i, we have the equation:
    # D_i - B0(t_i)P0 - B3(t_i)P3 ≈ B1(t_i)P1 + B2(t_i)P2
    # This is a linear system Ax = b, where x = [P1, P2]^T.

    # Bernstein basis polynomials for a cubic curve.
    # We only need the ones for the unknown inner control points.
    b1 = 3 * ts * (1 - ts)**2
    b2 = 3 * ts**2 * (1 - ts)

    # The matrix A for the linear system.
    A = np.stack([b1, b2], axis=1)

    # The vector b (or matrix b, since our points are 2D).
    # This is the part of the points' positions not explained by P0 and P3.
    b0 = (1 - ts)**3
    b3 = ts**3
    
    residual_points = points - b0[:, None] * p0 - b3[:, None] * p3

    try:
        p1, p2 = np.linalg.lstsq(A, residual_points, rcond=None)[0]
    except np.linalg.LinAlgError:
        # Default to a straight line.
        return BezierCurve(np.stack([p0, p3], axis=1))
    
    # check for degenerate cubic
    if np.linalg.norm(p1-p0) < 2:
        c = p2
    elif np.linalg.norm(p2-p3) < 2:
        c = p1
    else:
        return BezierCurve(np.stack([p0, p1, p2, p3], axis=1))

    return BezierCurve(np.stack([p0, c, p3], axis=1))

def sample_bezier_slider(
    segments: list[list[Coordinate]],
    length: float,
) -> tuple[
    Float[np.ndarray, "T 2"],
    Float[np.ndarray, "T"],
]:
    ctrl_curves: list[list[Vec2]] = [ [ np.array(pt) for pt in seg ] for seg in segments ]

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
    else:
        # shorter than defined, shorten last segment
        while path_length - length >= curves[-1].length:
            # slider doesn't reach last segment - shouldn't happen via editor, but possible to encode
            path_length -= curves.pop().length
        curves[-1], _ = curves[-1].split_at( 1 - (path_length - length) / curves[-1].length )
        
    cum_t = np.cumsum([ c.length for c in curves ])
    length = cum_t[-1]
    ts = np.linspace(0, length, int(max(4, length)))
    idxs = np.searchsorted(cum_t, ts)
    range_start = np.insert(cum_t, 0, 0)[idxs]
    rts = (ts - range_start) / (cum_t[idxs] - range_start)

    return np.stack([
        curves[idx].at(np.array([t]))[:,0]
        for idx, t in zip(idxs, rts)
    ], axis=0), ts / length