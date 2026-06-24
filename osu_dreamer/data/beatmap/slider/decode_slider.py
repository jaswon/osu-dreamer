

from jaxtyping import Float, Int

from numpy import ndarray

from .fit_bezier import fit_bezier_segment, fit_poly_bezier
from .fit_arc import fit_arc
from .slider_prior import (
    log_prior_arc,
    log_prior_single_bezier,
    log_prior_poly,
)

# expected cursor noise (in osu! pixels) used as the likelihood scale when choosing a slider type
# larger => trust the prior more (simpler sliders), smaller => fit more tightly.
NOISE_SCALE = 16.

# largest single bezier / poly-bezier complexity to consider as a candidate
MAX_BEZIER_CTRL = 8
MAX_SEGMENTS = 16


def decode_slider(
    cursor_signal: Float[ndarray, "2 L"], 
    start_idx: int, 
    end_idx: int, 
    num_repeats: int,
    noise_scale: float = NOISE_SCALE,
) -> tuple[str, float, list[Int[ndarray, "2"]]]:
    """
    returns the slider's curve type, length and control points defined by the
    cursor signal, start+end indices, and number of repeats.

    instead of fitting curves of increasing complexity until the error drops
    below a hand-tuned threshold, every candidate curve is scored by a MAP
    objective `sse / (2 * noise_scale**2) - log P(type)`, where `P(type)` is the
    empirical frequency of that slider type in the dataset (see `slider_prior`).
    this biases the decoder towards the simple slider types that dominate real maps.

    candidates are visited cheapest-prior first. since `sse >= 0`, a candidate's
    cost is bounded below by its prior penalty `-log P(type)`, so once that penalty
    exceeds the best cost found so far, no remaining (more complex) candidate can
    win and the search short-circuits -- this yields the same result as scoring
    every candidate, but usually only fits a handful of them.
    """

    first_slide_idx = round(start_idx + (end_idx-start_idx) / num_repeats)

    points = cursor_signal[:,start_idx:first_slide_idx+1]
    L = points.shape[1]
    if L < 2:
        # degenerate -- caller turns a zero-length slider into a hit circle
        return "B", 0., []

    inv_2var = 1 / (2 * noise_scale ** 2)

    def to_ctrl_pts(curves: list) -> list[Int[ndarray, "2"]]:
        # concatenating joined segments reproduces osu!'s repeated-point markers
        ctrl_pts: list[Int[ndarray, "2"]] = []
        for curve in curves:
            ctrl_pts.extend(curve.p.T.round().astype(int))
        return ctrl_pts

    # best so far: (cost, curve_type, length, ctrl_pts)
    best: tuple[float, str, float, list[Int[ndarray, "2"]]] | None = None

    def consider(cost: float, curve_type: str, length: float, ctrl_pts: list):
        nonlocal best
        if best is None or cost < best[0]:
            best = (cost, curve_type, length, ctrl_pts)

    # perfect arc
    arc = fit_arc(points)
    if arc is not None:
        sse, length, ctrl_pts = arc
        consider(sse * inv_2var - log_prior_arc(), "P", length, ctrl_pts)

    # single bezier candidates, visited in order of increasing prior penalty
    # (the penalty is known before fitting and is *not* monotonic in degree --
    # the cubic spike is cheaper than a quadratic -- so we sort rather than assume)
    single = sorted(
        (-log_prior_single_bezier(n), n)
        for n in range(2, min(MAX_BEZIER_CTRL, L) + 1)
    )
    for penalty, n_ctrl in single:
        if best is not None and penalty >= best[0]:
            # every remaining single bezier has at least this penalty -- none can win
            break
        curve, sse = fit_bezier_segment(points, n_ctrl)
        consider(sse * inv_2var + penalty, "B", curve.length, to_ctrl_pts([curve]))

    # poly-line and poly-bezier of increasing segment count
    for n_ctrl, all_lines in ((2, True), (4, False)):
        if best is not None and -log_prior_poly(2, all_lines) >= best[0]:
            # cheapest member of this family already can't win -- skip it entirely
            continue
        for curves, sse in fit_poly_bezier(points, n_ctrl, MAX_SEGMENTS):
            m = len(curves)
            consider(sse * inv_2var - log_prior_poly(m, all_lines), "B",
                     sum(c.length for c in curves), to_ctrl_pts(curves))
            # penalty only grows with segment count -- stop once it can't win
            if best is not None and -log_prior_poly(m + 1, all_lines) >= best[0]:
                break

    assert best is not None
    _, curve_type, length, ctrl_pts = best
    return curve_type, length, ctrl_pts