"""
Empirical prior over slider types, measured over the dataset.

The decoder uses this as the prior term of a MAP estimate when choosing which
kind of curve to emit for a slider, so that the choice is biased towards the
slider types that actually occur in real maps.

Slider taxonomy:
    P       perfect circle arc
    L       single straight line   (1 bezier segment,  2 control points)
    B/n     single bezier          (1 bezier segment,  n control points, n >= 3)
    PL/m    poly-line              (m line segments,   m >= 2)
    PB/m    poly-bezier            (m bezier segments, m >= 2)
"""

from math import log, exp

from scipy.special import zeta

# top-level slider-type counts measured over the dataset
TYPE_COUNTS: dict[str, int] = {
    "P":  938141,
    "L":  672639,
    "PL": 165541,
    "PB":  73003,
    "B":   55957,
}

# the number of segments in a poly-line / poly-bezier follows a zeta (discrete
# power-law) distribution `P(m) ~ m ** -ZETA_S` over `m >= 2`
ZETA_S = 4

# a single bezier's control-point count is bimodal -- a sharp spike at the cubic
# (the canonical bezier osu! editors emit) over a geometric tail -- so it is
# modelled as a two-component mixture over `n_ctrl >= 3`:
#     P(n) = w * [n == 4] + (1 - w) * (1 - q) * q ** (n - 3)
B_CUBIC_WEIGHT = 0.5991  # mixture weight `w` of the cubic spike
B_TAIL_DECAY = 0.7431    # geometric ratio `q` of the degree tail

_TOTAL = sum(TYPE_COUNTS.values())
_LOG_TYPE = {k: log(v / _TOTAL) for k, v in TYPE_COUNTS.items()}


def _zeta_log_pmf(k: int, k_min: int) -> float:
    """log P(k) for a zeta distribution truncated to the support `k >= k_min`"""
    norm = zeta(ZETA_S) - sum(j ** -ZETA_S for j in range(1, k_min))
    return -ZETA_S * log(k) - log(norm)


def log_prior_arc() -> float:
    return _LOG_TYPE["P"]


def log_prior_single_bezier(n_ctrl: int) -> float:
    """log-prior of a single bezier segment with `n_ctrl` control points"""
    if n_ctrl <= 2:
        # degree 1 == straight line
        return _LOG_TYPE["L"]
    # cubic-spike + geometric-tail mixture over the control-point count (n >= 3),
    # evaluated in log-space so the geometric tail never underflows
    w, q = B_CUBIC_WEIGHT, B_TAIL_DECAY
    log_tail = log(1 - w) + log(1 - q) + (n_ctrl - 3) * log(q)
    if n_ctrl == 4:
        log_w = log(w)
        hi = max(log_w, log_tail)
        log_degree = hi + log(exp(log_w - hi) + exp(log_tail - hi))
    else:
        log_degree = log_tail
    return _LOG_TYPE["B"] + log_degree


def log_prior_poly(n_segments: int, all_lines: bool) -> float:
    """log-prior of a poly-line / poly-bezier with `n_segments` segments"""
    key = "PL" if all_lines else "PB"
    return _LOG_TYPE[key] + _zeta_log_pmf(n_segments, k_min=2)
