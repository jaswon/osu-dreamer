# https://github.com/volkerp/fitCurves

from jaxtyping import Float

from typing import Optional, Union
import numpy as np
from numpy import ndarray

import bezier

def hodo(p: Float[ndarray, "N 2"]):
    return p.shape[0] * (p[1:] - p[:-1])

def q(p: Float[ndarray, "N 2"], t: Float[ndarray, "L"]) -> Float[ndarray, "L 2"]:
    """evaluates bezier at t"""
    return bezier.Curve.from_nodes(p.T).evaluate_multi(t).T

def qprime(p: Float[ndarray, "N 2"], t: Float[ndarray, "L"]) -> Float[ndarray, "L 2"]:
    """evaluates bezier first derivative at t"""
    return bezier.Curve.from_nodes(hodo(p).T).evaluate_multi(t).T

def qprimeprime(p: Float[ndarray, "N 2"], t: Float[ndarray, "L"]) -> Float[ndarray, "L 2"]:
    """evaluates bezier second derivative at t"""
    return bezier.Curve.from_nodes(hodo(hodo(p)).T).evaluate_multi(t).T

def normalize(v: Float[ndarray, "..."]) -> Float[ndarray, "..."]:
    magnitude = np.sqrt(np.dot(v,v))
    if magnitude < np.finfo(float).eps:
        return v
    return v / magnitude

def compute_error(
    p: Float[ndarray, "N 2"], 
    points: Float[ndarray, "L 2"], 
    u: Float[ndarray, "L"],
) -> tuple[float, int]:
    errs = ((q(p, u) - points) ** 2).sum(-1) # L
    split_point = errs.argmax()
    return float(errs[split_point]), int(split_point)

Point = Float[ndarray, "2"]
Cubic = Float[ndarray, "4 2"]
Line = Float[ndarray, "2 2"]
Segment = Union[Cubic, Line]

def segment_length(p: Segment) -> float:
    return float(bezier.Curve.from_nodes(p.T).length)

def fit_bezier(
    points: Float[ndarray, "L 2"],
    max_err: float, 
    left_tangent: Optional[Point] = None, # [2]
    right_tangent: Optional[Point] = None, # [2]
) -> list[Segment]:
    """fit one (or more) Bezier curves to a set of points"""

    if points.shape[0] < 2:
        # cannot fit bezier to a single point
        return []
    
    weights = (lambda x,n: (x**-np.arange(1,n+1)) / (1 - x**-n) * (x-1))(2., min(5, len(points)-2)) # N
    
    if left_tangent is None:
        # points[1] - points[0]
        l_vecs = points[2:2+len(weights)] - points[1] # N 2
        left_tangent = normalize(np.einsum('np,n->p', l_vecs, weights))
        
    if right_tangent is None:
        # points[-2] - points[-1]
        r_vecs = points[-3:-3-len(weights):-1] - points[-2] # N 2
        right_tangent = normalize(np.einsum('np,n->p', r_vecs, weights))
    
    if points.shape[0] == 2:
        return [points]
    
    # parameterize points, assuming constant speed
    u = np.cumsum(np.linalg.norm(points[1:] - points[:-1], axis=1))
    if u[-1] == 0:
        # no distance covered
        return []
    u = np.pad(u, (1,0)) / u[-1]

    split_point = points.shape[0] // 2 # makes type checker happy
    for _ in range(32):
        bez_curve = generate_bezier(points, u, left_tangent, right_tangent)
        err, split_point = compute_error(bez_curve, points, u)
            
        if err < max_err:
            # check if line is a good fit
            line_err, _ = compute_error(bez_curve[[0,-1]], points, u)
            if line_err < max_err:
                return [bez_curve[[0,-1]]]

            return [bez_curve]
        
        # iterate parameterization
        u = newton_raphson_root_find(bez_curve, points, u)

    # Fitting failed -- split at max error point and fit recursively
    center_tangent = normalize(points[split_point-1] - points[split_point+1])
    return [
        *fit_bezier(points[:split_point+1], max_err, left_tangent, center_tangent),
        *fit_bezier(points[split_point:], max_err, -center_tangent, right_tangent),
    ]

def generate_bezier(
    points: Float[ndarray, "L 2"],      # [L,2]
    u: Float[ndarray, "L"],             # [L]
    left_tangent: Point,  # [2]
    right_tangent: Point, # [2]
) -> Cubic:
    bez_curve = np.array([points[0], points[0], points[-1], points[-1]]) # 4 2

    # compute the A's
    A = (3 * (1-u) * u * np.array([1-u,u])).T[..., None] * np.array([left_tangent, right_tangent])
    
    # Create the C and X matrices
    C = np.einsum('lix,ljx->ij', A, A)
    X = np.einsum('lix,lx->i', A, points - q(bez_curve, u))

    # Compute the determinants of C and X
    det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1]
    det_C0_X  = C[0][0] * X[1]    - C[1][0] * X[0]
    det_X_C1  = X[0]    * C[1][1] - X[1]    * C[0][1]

    # Finally, derive alpha values
    alpha_l = 0.0 if abs(det_C0_C1) < 1e-5 else det_X_C1 / det_C0_C1
    alpha_r = 0.0 if abs(det_C0_C1) < 1e-5 else det_C0_X / det_C0_C1

    # If alpha negative, use the Wu/Barsky heuristic (see text)
    # (if alpha is 0, you get coincident control points that lead to
    # divide by zero in any subsequent NewtonRaphsonRootFind() call)
    seg_len = np.linalg.norm(points[0] - points[-1])
    epsilon = 1e-6 * seg_len
    if alpha_l < epsilon or alpha_r < epsilon:
        # fall back on standard (probably inaccurate) formula, and subdivide further if needed.
        bez_curve[1] += left_tangent * (seg_len / 3.0)
        bez_curve[2] += right_tangent * (seg_len / 3.0)

    else:
        # First and last control points of the Bezier curve are
        # positioned exactly at the first and last data points
        # Control points 1 and 2 are positioned an alpha distance out
        # on the tangent vectors, left and right, respectively
        bez_curve[1] += left_tangent * alpha_l
        bez_curve[2] += right_tangent * alpha_r

    return bez_curve


def newton_raphson_root_find(
    bez: Float[ndarray, "N 2"], 
    points: Float[ndarray, "L 2"], 
    u: Float[ndarray, "L"],
) -> Float[ndarray, "L"]:
    """
    Newton's root finding algorithm calculates f(x)=0 by reiterating
    x_n+1 = x_n - f(x_n)/f'(x_n)
    We are trying to find curve parameter u for some point p that minimizes
    the distance from that point to the curve. Distance point to curve is d=q(u)-p.
    At minimum distance the point is perpendicular to the curve.
    We are solving
    f = q(u)-p * q'(u) = 0
    with
    f' = q'(u) * q'(u) + q(u)-p * q''(u)
    gives
    u_n+1 = u_n - |q(u_n)-p * q'(u_n)| / |q'(u_n)**2 + q(u_n)-p * q''(u_n)|
    """
    
    d = q(bez, u) - points # L 2
    qp = qprime(bez, u) # L 2
    num = (d * qp).sum(-1) # L
    den = (qp**2 + d*qprimeprime(bez, u)).sum(-1) # L
    
    return u - np.divide(num, den, out=np.zeros_like(num), where=den!=0)