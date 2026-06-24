
from typing import Iterator

from jaxtyping import Float

import numpy as np
from numpy import ndarray
from scipy.special import comb

from osu_dreamer.osu.bezier import BezierCurve


def _bernstein(t: Float[ndarray, "L"], n_ctrl: int) -> Float[ndarray, "L n_ctrl"]:
    """Bernstein basis matrix mapping `n_ctrl` control points to curve points"""
    d = n_ctrl - 1
    i = np.arange(n_ctrl)
    return comb(d, i) * t[:, None] ** i * (1 - t[:, None]) ** (d - i)


def fit_bezier_segment(
    points: Float[ndarray, "2 L"],
    n_ctrl: int,
    constrain_start: bool = False,
    constrain_end: bool = False,
) -> tuple[BezierCurve, float]:
    """
    fits a single bezier with `n_ctrl` control points to points evenly spaced in
    time (via linearly constrained least squares), returning the curve and its
    sum of squared residuals.

    when `constrain_start` / `constrain_end` is set the corresponding endpoint is
    pinned to the data endpoint (used to keep adjacent poly-segments joined).
    """

    L = points.shape[1]
    t = np.linspace(0, 1, L)
    TM = _bernstein(t, n_ctrl)  # L x n_ctrl

    # initial guess: control points linearly interpolated between the endpoints
    c0 = np.linspace(points[:, 0], points[:, -1], n_ctrl)  # n_ctrl x 2

    free = np.ones(n_ctrl, dtype=bool)
    if constrain_start:
        free[0] = False
    if constrain_end:
        free[-1] = False
    N = np.eye(n_ctrl)[:, free]  # n_ctrl x f

    A = TM @ N  # L x f
    y = points.T - TM @ c0  # L x 2
    try:
        z = np.linalg.solve(A.T @ A, A.T @ y)
    except np.linalg.LinAlgError:
        # singular => infinitely many solutions, pseudoinverse yields minimal one
        z = np.linalg.pinv(A) @ y
    P = c0 + N @ z  # n_ctrl x 2

    sse = float(((TM @ P - points.T) ** 2).sum())
    return BezierCurve(P.T), sse


def fit_poly_bezier(
    points: Float[ndarray, "2 L"],
    n_ctrl: int,
    max_segments: int,
) -> Iterator[tuple[list[BezierCurve], float]]:
    """
    yields poly-bezier fits with increasing segment count (2 .. max_segments),
    where every segment has `n_ctrl` control points (use 2 for a poly-line).

    segments are grown greedily by repeatedly splitting the worst-fitting segment
    at its point of largest error
    """

    L = points.shape[1]

    def fit(lo: int, hi: int) -> tuple[BezierCurve, float, int]:
        seg = points[:, lo:hi + 1]
        curve, sse = fit_bezier_segment(
            seg, min(n_ctrl, seg.shape[1]),
            constrain_start=lo != 0,
            constrain_end=hi != L - 1,
        )
        t = np.linspace(0, 1, seg.shape[1])
        resid = ((curve.at(t) - seg) ** 2).sum(0)
        return curve, sse, lo + int(resid.argmax())

    spans = [(0, L - 1)]
    fits = [fit(0, L - 1)]

    for _ in range(2, max_segments + 1):
        # split the worst-fitting span that still contains an interior point
        candidates = [k for k, (lo, hi) in enumerate(spans) if hi - lo >= 2]
        if not candidates:
            break
        k = max(candidates, key=lambda k: fits[k][1])
        lo, hi = spans[k]
        split = fits[k][2]
        if not lo < split < hi:
            split = (lo + hi) // 2

        spans = spans[:k] + [(lo, split), (split, hi)] + spans[k + 1:]
        fits = fits[:k] + [fit(lo, split), fit(split, hi)] + fits[k + 1:]

        yield [f[0] for f in fits], float(sum(f[1] for f in fits))