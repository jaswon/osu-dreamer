
import numpy as np

from osu_dreamer.osu.bezier import BezierCurve

def reduce_to_poly_cubic(
    curve: BezierCurve,
    max_allowed_err: float = 10., 
) -> list[BezierCurve]:
    """reduces a high-order Bezier curve to a cubic poly-Bezier curve"""

    assert curve.degree >= 4
    if curve.degree == 4:
        return [curve]
    
    cubic = reduce_to_cubic(curve)

    # find approximate max error
    u = np.linspace(0, 1, 3*curve.degree)
    errs = np.linalg.norm(cubic.at(u) - curve.at(u), axis=0) # L
    max_err_i = errs.argmax()

    if errs[max_err_i] < max_allowed_err:
        return [cubic]
    
    # split at max error location and fit recursively
    split_a, split_b = curve.split_at(t=float(u[max_err_i]))
    return [
        *reduce_to_poly_cubic(split_a, max_allowed_err),
        *reduce_to_poly_cubic(split_b, max_allowed_err),
    ]

def reduce_to_cubic(curve: BezierCurve) -> BezierCurve:
    P, D = curve.p.T, curve.degree
    
    # endpoints
    q0 = P[0]
    q3 = P[D-1]

    # endpoint derivatives
    d0 = D * (P[1] - q0)
    d1 = D * (q3 - P[D-2])

    t = np.linspace(0, 1, 3*D)

    # Bernstein basis (vectorized)
    b0 = (1 - t) ** 3
    b1 = 3 * (1 - t) ** 2 * t
    b2 = 3 * (1 - t) * t ** 2
    b3 = t ** 3

    C = np.outer(b0+b1, q0) + np.outer(b2+b3, q3)
    Y = curve.at(t).T - C

    A = np.outer(b1, d0)
    B = np.outer(-b2, d1)

    alpha, beta = np.linalg.lstsq(
        np.column_stack([A.ravel(), B.ravel()]), 
        Y.ravel(), 
        rcond=None,
    )[0]

    q1 = q0 + d0 * alpha
    q2 = q3 - d1 * beta

    # check for degenerate cubic
    if np.linalg.norm(q1-q0) < 2:
        c = q2
    elif np.linalg.norm(q2-q3) < 2:
        c = q1
    else:
        return BezierCurve(np.array([q0,q1,q2,q3]).T)

    q1 = q0 + (c-q0) * 2/3
    q2 = q3 + (c-q3) * 2/3

    return BezierCurve(np.array([q0,q1,q2,q3]).T)