"""Piecewise-linear functions on [0,1], hat functions and envelopes.

This module is the computational core of the library.  Every metric parameter
of a continuous graph reduces to the same object: the eccentricity restricted
to one edge is a piecewise-linear function of θ, and it is the *upper envelope*
of m+1 **hat functions**, one per edge of the graph (the edge itself
contributing two degenerate hats).  Once envelopes are
available:

* the diameter is the largest ceiling of any hat;
* the eccentricity of a point is one evaluation;
* the local radius and local center of an edge are the minimum of an envelope
  and the set where it is attained;
* the radius and the center aggregate those over the edges.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple

__all__ = ["HatFunction", "PiecewiseLinear", "upper_envelope"]


@dataclass(frozen=True)
class HatFunction:
    """A hat function on [0,1].

    The function rises with slope ``slope`` up to θ = b, stays at the
    *ceiling* ``h`` on [b, s], and falls with slope ``-slope`` afterwards:

                   ⎧ h - (b - θ) L    if 0 ≤ θ ≤ b,
        hat(θ) =   ⎨ h                if b ≤ θ ≤ s,
                   ⎩ h - (θ - s) L    if s ≤ θ ≤ 1.

    A hat is degenerate when a sloped piece is missing (``b == 0`` or
    ``s == 1``) or the interval [b, s] reduces to a point (``b == s``).  Degenerate
    hats need no special treatment, which is why the valley (4.14) of an edge
    against itself is represented as the envelope of two degenerate hats.

    Attributes
    ----------
    b, s : float
        Endpoints of the interval [b, s] where the hat attains its ceiling,
        with ``0 <= b <= s <= 1``.
    h : float
        Ceiling, i.e. the maximum of the function.
    slope : float
        Absolute value of the slope of the two sloped branches; for the
        eccentricity along an edge this is the edge length.
    """

    b: float
    s: float
    h: Any
    slope: Any

    def __post_init__(self) -> None:
        if not (0 <= self.b <= self.s <= 1):
            raise ValueError(
                f"a hat needs 0 <= b <= s <= 1, got b={self.b!r}, s={self.s!r}"
            )

    def __call__(self, theta: float):
        if theta < self.b:
            return self.h - (self.b - theta) * self.slope
        if theta > self.s:
            return self.h - (theta - self.s) * self.slope
        return self.h

    def breakpoints(self) -> "PiecewiseLinear":
        """Return the hat as a :class:`PiecewiseLinear` function."""
        zero = self.b - self.b      # keep the numeric type of the data
        xs = [zero, self.b, self.s, zero + 1]
        ys = [self(x) for x in xs]
        return PiecewiseLinear(xs, ys)


class PiecewiseLinear:
    """A continuous piecewise-linear function on [0,1].

    The function is stored as an increasing list of breakpoints ``xs`` (with
    ``xs[0] == 0`` and ``xs[-1] == 1``) and the corresponding values ``ys``; it
    is linear between consecutive breakpoints.

    Examples
    --------
    >>> f = PiecewiseLinear([0.0, 0.5, 1.0], [1.0, 0.0, 1.0])
    >>> f(0.25)
    0.5
    >>> value, where = f.minimum()
    >>> value, where
    (0.0, [(0.5, 0.5)])
    """

    __slots__ = ("xs", "ys")

    def __init__(self, xs: Sequence[float], ys: Sequence[Any]) -> None:
        if len(xs) != len(ys) or len(xs) < 2:
            raise ValueError("xs and ys must have the same length, at least 2")
        clean_xs: List[float] = []
        clean_ys: List[Any] = []
        for x, y in zip(xs, ys):
            if clean_xs and x == clean_xs[-1]:
                # Duplicated abscissa: keep the larger value, so that merging
                # never lowers an envelope.
                if y > clean_ys[-1]:
                    clean_ys[-1] = y
                continue
            if clean_xs and x < clean_xs[-1]:
                raise ValueError("xs must be non-decreasing")
            clean_xs.append(x)
            clean_ys.append(y)
        self.xs = clean_xs
        self.ys = clean_ys

    # -- evaluation -----------------------------------------------------

    def __call__(self, theta: float):
        xs, ys = self.xs, self.ys
        if theta <= xs[0]:
            return ys[0]
        if theta >= xs[-1]:
            return ys[-1]
        i = bisect_right(xs, theta) - 1
        if xs[i] == theta:
            return ys[i]
        x0, x1 = xs[i], xs[i + 1]
        y0, y1 = ys[i], ys[i + 1]
        return y0 + (y1 - y0) * (theta - x0) / (x1 - x0)

    def __len__(self) -> int:
        return len(self.xs)

    def points(self) -> List[Tuple[float, Any]]:
        """Return the list of ``(theta, value)`` breakpoints."""
        return list(zip(self.xs, self.ys))

    # -- extrema --------------------------------------------------------

    def maximum(self):
        """Return the maximum value of the function."""
        return max(self.ys)

    def minimum(self, tol: float = 0.0) -> Tuple[Any, List[Tuple[float, float]]]:
        """Return ``(value, intervals)``: the minimum and where it is attained.

        A piecewise-linear function attains its minimum at a breakpoint, and
        the set where it does so is a finite union of closed intervals whose
        endpoints are breakpoints.  ``intervals`` is that union, as a list of
        ``(theta_low, theta_high)`` pairs; a single point is reported as a
        degenerate interval ``(t, t)``.

        Parameters
        ----------
        tol : float
            Values within ``tol`` of the minimum count as the minimum.  Use
            ``0`` with exact arithmetic.
        """
        best = min(self.ys)
        flags = [y <= best + tol for y in self.ys]
        intervals: List[Tuple[float, float]] = []
        i = 0
        n = len(flags)
        while i < n:
            if not flags[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            intervals.append((self.xs[i], self.xs[j]))
            i = j + 1
        return best, intervals


def upper_envelope(hats: Iterable[HatFunction]) -> PiecewiseLinear:
    """Return the upper envelope max_k hat_k of the hats.

    Implements Algorithm 1 of Hansen, Labbé and Nicolas (Section 4.4.3 of the
    master's thesis): the hats are inserted in non-decreasing order of
    ceiling, which makes each insertion local.  Since every hat is bounded
    above by its own ceiling, the envelope of the first k hats never exceeds
    h_k ≤ h_{k+1}; therefore the incoming hat dominates the envelope on the
    whole interval [b, s] where it attains its ceiling, and, because the
    difference *incoming minus envelope* is monotone on each side of [b, s]
    (its derivative lies in {0, L, 2L}), it dominates exactly on a single
    interval [t_l, t_r] ⊇ [b, s].
    Updating therefore amounts to finding two crossings and splicing.

    Walking outwards from [b, s] to find each crossing only ever visits
    breakpoints that are about to be deleted, plus one; since a breakpoint is
    created once and deleted at most once, the walks cost O(k) in total.
    The breakpoints, however, are stored in a Python list, which is rebuilt at
    each insertion: an insertion costs O(k), and the envelope of k hats
    O(k²).  Storing them in an ordered structure with O(log k) insertion and
    deletion, such as the 2–3 tree of the original algorithm, would bring the
    total down to O(k log k), the cost of the initial sort.

    Raises
    ------
    ValueError
        If ``hats`` is empty, or if the hats do not share a common slope (the
        argument above relies on all sloped branches having slope ±L).
    """
    hats = sorted(hats, key=lambda hat: hat.h)
    if not hats:
        raise ValueError("at least one hat function is required")
    slope = hats[0].slope
    if any(hat.slope != slope for hat in hats):
        raise ValueError("all hat functions must share the same slope")

    first = hats[0]
    # Derive 0 and 1 from the data, so that exact (Fraction) inputs stay exact.
    zero = (first.b + first.s) * 0
    xs: List[float] = [zero, first.b, first.s, zero + 1]
    ys: List[Any] = [first(x) for x in xs]
    xs, ys = _dedup(xs, ys)

    for hat in hats[1:]:
        xs, ys = _insert(xs, ys, hat)
    return PiecewiseLinear(xs, ys)


def _dedup(xs: List[float], ys: List[Any]) -> Tuple[List[float], List[Any]]:
    out_x: List[float] = []
    out_y: List[Any] = []
    for x, y in zip(xs, ys):
        if out_x and x == out_x[-1]:
            if y > out_y[-1]:
                out_y[-1] = y
            continue
        out_x.append(x)
        out_y.append(y)
    return out_x, out_y


def _eval(xs: List[float], ys: List[Any], theta: float):
    if theta <= xs[0]:
        return ys[0]
    if theta >= xs[-1]:
        return ys[-1]
    i = bisect_right(xs, theta) - 1
    if xs[i] == theta:
        return ys[i]
    x0, x1 = xs[i], xs[i + 1]
    y0, y1 = ys[i], ys[i + 1]
    return y0 + (y1 - y0) * (theta - x0) / (x1 - x0)


def _cross(x0: float, y0: Any, x1: float, y1: Any, line_x: float, line_y: Any,
           slope: Any) -> float:
    """Abscissa where the segment ``(x0,y0)-(x1,y1)`` meets the line of given
    slope through ``(line_x, line_y)``, clipped to ``[x0, x1]``."""
    if x1 == x0:
        return x0
    seg_slope = (y1 - y0) / (x1 - x0)
    denom = seg_slope - slope
    if denom == 0:
        return x0
    t = (line_y - slope * line_x - y0 + seg_slope * x0) / denom
    if t < x0:
        return x0
    if t > x1:
        return x1
    return t


def _insert(xs: List[float], ys: List[Any], hat: HatFunction):
    """Splice ``hat`` into the envelope given by ``xs``/``ys``."""
    b, s, h, L = hat.b, hat.s, hat.h, hat.slope

    # -- left crossing: the rising branch, of slope +L through (b, h) -------
    i = bisect_right(xs, b) - 1
    j = i
    while j >= 0 and h - (b - xs[j]) * L >= ys[j]:
        j -= 1
    if j < 0:
        t_left = xs[0]
        left_xs: List[float] = []
        left_ys: List[Any] = []
    else:
        x0, y0 = xs[j], ys[j]
        if j + 1 <= i:
            x1, y1 = xs[j + 1], ys[j + 1]
        else:  # b lies strictly inside the segment (xs[j], xs[j + 1])
            x1, y1 = b, _eval(xs, ys, b)
        t_left = _cross(x0, y0, x1, y1, b, h, L)
        left_xs = xs[: j + 1]
        left_ys = ys[: j + 1]

    # -- right crossing: the falling branch, of slope -L through (s, h) -----
    n = len(xs)
    k0 = bisect_left(xs, s)
    k = k0
    while k < n and h - (xs[k] - s) * L >= ys[k]:
        k += 1
    if k >= n:
        t_right = xs[-1]
        right_xs: List[float] = []
        right_ys: List[Any] = []
    else:
        x1, y1 = xs[k], ys[k]
        if k - 1 >= k0:
            x0, y0 = xs[k - 1], ys[k - 1]
        else:  # s lies strictly inside the segment (xs[k - 1], xs[k])
            x0, y0 = s, _eval(xs, ys, s)
        t_right = _cross(x0, y0, x1, y1, s, h, -L)
        right_xs = xs[k:]
        right_ys = ys[k:]

    mid_xs = [t_left, b, s, t_right]
    mid_ys = [h - (b - t_left) * L, h, h, h - (t_right - s) * L]

    return _dedup(left_xs + mid_xs + right_xs, left_ys + mid_ys + right_ys)
