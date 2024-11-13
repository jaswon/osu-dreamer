
from jaxtyping import Float

import numpy as np
from numpy import ndarray

from osu_dreamer.osu.bezier import BezierCurve

def fit_bezier(
    points: Float[ndarray, "2 L"],
    max_err: float, 
    constrain_start: bool = False,
    constrain_end: bool = False,
) -> list[BezierCurve]:
    """fits a cubic poly-Bezier curve to points evenly spaced in time"""

    L = points.shape[1]
    if L < 2:
        # cannot fit bezier to a single point
        return []
    
    if L == 2:
        # line
        return [BezierCurve(points)]

    u = np.linspace(0, 1, L)
    bez_curve = fit_cubic(points, u, constrain_start, constrain_end)
    err, split_point = compute_error(bez_curve, points, u)
        
    if err < max_err:
        # check if line is a good fit
        bez_line = bez_curve[:, [0,-1]]
        line_err, _ = compute_error(bez_line, points, u)
        if line_err < max_err:
            return [BezierCurve(bez_line)]

        return [BezierCurve(bez_curve)]

    # Fitting failed -- split at max error point and fit recursively
    return [
        *fit_bezier(points[:,:split_point+1], max_err, constrain_start, True),
        *fit_bezier(points[:,split_point:], max_err, True, constrain_end),
    ]

def compute_error(
    p: Float[ndarray, "2 N"], 
    points: Float[ndarray, "2 L"], 
    u: Float[ndarray, "L"],
) -> tuple[float, int]:
    errs = ((BezierCurve(p).at(u) - points) ** 2).sum(0) # L
    split_point = errs.argmax()
    return errs[split_point], int(split_point)

def fit_cubic(
    xy: Float[ndarray, "2 L"],
    t: Float[ndarray, "L"],
    constrain_start: bool,
    constrain_end: bool,
) -> Float[ndarray, "2 4"]:
    """
    fits a cubic bezier constrained by start and end points (via linearly constrained least squares) 
    """

    TM = t[:,None] ** np.array([[0,1,2,3]]) @ np.array([
        [1, 0, 0, 0],
        [-3, 3, 0, 0],
        [3, -6, 3, 0],
        [-1, 3, -3, 1],
    ])

    c0 = np.array([
        xy[:,0], 
        xy[:,0] + (xy[:,-1] - xy[:,0]) / 3, 
        xy[:,-1] + (xy[:,0] - xy[:,-1]) / 3, 
        xy[:,-1],
    ])
    if constrain_start and constrain_end:
        N = np.eye(4,2,-1)
    elif constrain_start and not constrain_end:
        N = np.eye(4,3,-1)
    elif not constrain_start and constrain_end:
        N = np.eye(4,3,0)
    else:
        N = np.eye(4,4,0)
    
    A_tilde = TM @ N
    y_tilde = xy.T - TM @ c0
    try:
        z_hat = np.linalg.inv(A_tilde.T @ A_tilde) @ A_tilde.T
    except:
        # singular => infinitely many solutions
        # pseudoinverse yields minimal solution
        z_hat = np.linalg.pinv(A_tilde)
    X_hat = c0 + N @ z_hat @ y_tilde

    return X_hat.T