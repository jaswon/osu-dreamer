
from jaxtyping import Float, Int

import numpy as np
from numpy import ndarray

def fit_arc(
    points: Float[ndarray, "2 L"],
) -> tuple[float, float, list[Int[ndarray, "2"]]] | None:
    """
    fits an arc to points evenly spaced in time, returning its sum of squared
    residuals, length and control points, or None if the points do not form a
    valid (renderable) arc.
    """

    if points.shape[1] < 3:
        return None

    x, y = points
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x*x + y*y

    try:
        cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    r_squared = cx*cx + cy*cy + c
    if r_squared <= 0:
        return None

    center = np.array([cx, cy])
    radius = r_squared ** .5

    start = points[:,0]
    end = points[:,-1]
    if np.linalg.norm(end - start) < 15:
        return None

    if radius > 320:
        return None

    angles = np.unwrap(np.arctan2(y - cy, x - cx))
    sweep = angles[-1] - angles[0]
    if abs(sweep) < .05:
        return None

    deltas = np.diff(angles)
    if np.count_nonzero(np.sign(deltas) != np.sign(sweep)) > len(deltas) * .25:
        return None

    radial_err = np.linalg.norm(points - center[:,None], axis=0) - radius
    sse = float((radial_err**2).sum())

    mid_angle = angles[0] + sweep / 2
    mid = center + radius * np.array([np.cos(mid_angle), np.sin(mid_angle)])
    length = abs(sweep) * radius
    ctrl_pts = np.column_stack([start, mid, end]).T.round().astype(int)

    return sse, length, list(ctrl_pts)