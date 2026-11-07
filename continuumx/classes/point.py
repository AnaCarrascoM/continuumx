"""Points and segments of a continuous graph.

A point of the continuous graph 𝒢 defined by G = (V, E) with lengths ℓ is a
triple (uv, u, λ) with uv ∈ E and λ ∈ [0, ℓ(uv)].  Two triples describe the
same point whenever

    (uv, u, λ) ~ (vu, v, ℓ(uv) - λ),

and, in addition, every triple with λ ∈ {0, ℓ(uv)} describes a *vertex*, which
is shared by all the edges incident to it.

This module stores the normalized parameter θ = λ / ℓ(uv) ∈ [0, 1] rather
than λ, for two reasons:

* it is the parametrization used throughout Chapter 4 of the master's thesis,
so code and formulas read alike;
* it is intrinsic: a triple (u, v, θ) can be validated without knowing
ℓ(uv), whereas (u, v, λ) cannot.

The offset λ is recovered with
:meth:`~continuumx.classes.continuous_graph.ContinuousGraph.offset`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterator, Optional, Tuple

from continuumx.exceptions import InvalidPointError

__all__ = ["Point", "Segment", "node_key", "orient"]


def node_key(node: Hashable) -> Tuple[str, str]:
    """Return a total, deterministic sort key for an arbitrary graph node.

    NetworkX nodes are only required to be hashable, not orderable, and a
    graph may legitimately mix types (``0`` and ``"a"``).  Canonical
    orientation of an edge nevertheless needs some total order; the exact
    order is irrelevant as long as it is deterministic within a run.
    """
    return (type(node).__name__, repr(node))


def orient(u: Hashable, v: Hashable) -> Tuple[Hashable, Hashable, bool]:
    """Return ``(u, v, swapped)`` with the endpoints in canonical order."""
    if node_key(v) < node_key(u):
        return v, u, True
    return u, v, False


@dataclass(frozen=True)
class Point:
    """A point of a continuous graph, in canonical form.

    Parameters
    ----------
    u, v : hashable
        Endpoints of the edge containing the point.  Passing ``u == v``
        denotes the *vertex* ``u`` itself.
    theta : float
        Parameter θ ∈ [0, 1] of the point along the edge, measured from
        ``u``: ``theta == 0`` is ``u`` and ``theta == 1`` is ``v``.

    Notes
    -----
    Instances are normalized on construction, so that different triples
    describing the same point of 𝒢 compare equal and hash equally:

    * ``Point(v, u, t)`` becomes ``Point(u, v, 1 - t)`` when ``u`` precedes
      ``v`` in the canonical order given by :func:`node_key`;
    * ``Point(u, v, 0)`` and ``Point(u, w, 0)`` both become ``Point(u, u, 0)``,
      the vertex ``u``.

    Because the normalization computes ``1 - theta``, exact equality of
    ``float`` points is subject to the usual rounding: ``Point("b", "a", 0.3)``
    and ``Point("a", "b", 0.7)`` need not be identical bit for bit.  Use
    :meth:`~continuumx.classes.continuous_graph.ContinuousGraph.same_point` for
    a tolerant comparison, or build the graph with :class:`fractions.Fraction`
    lengths and parameters for exact arithmetic.

    Examples
    --------
    >>> Point("b", "a", 0.25)
    Point('a', 'b', 0.75)
    >>> Point("a", "b", 0.0)
    Point('a')
    >>> Point.at_vertex("c").is_vertex
    True
    """

    u: Hashable
    v: Hashable
    theta: float

    def __post_init__(self) -> None:
        theta = self.theta
        # 0 with the numeric type of theta (exact for Fraction)
        zero = theta * 0
        if theta < zero or theta > zero + 1:
            raise InvalidPointError(
                f"theta must lie in [0, 1], got {theta!r}. "
                "theta is the normalized position along the edge; use "
                "ContinuousGraph.point(u, v, offset=...) to build a point from "
                "its distance to u."
            )
        if self.u == self.v:
            object.__setattr__(self, "theta", zero)
            return
        if theta == zero:
            object.__setattr__(self, "v", self.u)
            object.__setattr__(self, "theta", zero)
            return
        if theta == zero + 1:
            object.__setattr__(self, "u", self.v)
            object.__setattr__(self, "theta", zero)
            return
        u, v, swapped = orient(self.u, self.v)
        if swapped:
            object.__setattr__(self, "u", u)
            object.__setattr__(self, "v", v)
            object.__setattr__(self, "theta", (zero + 1) - theta)

    # -- constructors ---------------------------------------------------

    @classmethod
    def at_vertex(cls, w: Hashable) -> "Point":
        """Return the point of 𝒢 that is the vertex ``w``."""
        return cls(w, w, 0.0)

    # -- queries --------------------------------------------------------

    @property
    def is_vertex(self) -> bool:
        """Whether the point is a vertex of the graph."""
        return self.u == self.v

    @property
    def vertex(self) -> Optional[Hashable]:
        """The vertex this point is, or ``None`` if it is interior to an edge."""
        return self.u if self.is_vertex else None

    @property
    def edge(self) -> Optional[Tuple[Hashable, Hashable]]:
        """The edge containing the point, or ``None`` for a vertex.

        A vertex belongs to every incident edge, so no single edge can be
        returned for it; that is why vertices are normalized to ``(w, w, 0)``.
        """
        return None if self.is_vertex else (self.u, self.v)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if self.is_vertex:
            return f"Point({self.u!r})"
        return f"Point({self.u!r}, {self.v!r}, {self.theta!r})"


@dataclass(frozen=True)
class Segment:
    """A closed sub-segment of a single edge of a continuous graph.

    The continuous center is in general infinite, a union of stretches of
    edges, so the library needs a type for it.  A :class:`Segment` is the
    portion of the edge ``uv`` between ``theta_min`` and ``theta_max``.  A
    degenerate segment (``theta_min == theta_max``) is a single point.

    Examples
    --------
    >>> Segment("a", "b", 0.25, 0.75).length_fraction
    0.5
    >>> Segment("a", "b", 0.5, 0.5).is_point
    True
    """

    u: Hashable
    v: Hashable
    theta_min: float
    theta_max: float

    def __post_init__(self) -> None:
        lo, hi = self.theta_min, self.theta_max
        # 0 with the numeric type of the parameters (exact for Fraction)
        zero = lo * 0
        if lo > hi:
            lo, hi = hi, lo
        if lo < zero or hi > zero + 1:
            raise InvalidPointError(
                f"segment parameters must lie in [0, 1], got ({lo!r}, {hi!r})"
            )
        if self.u == self.v:
            object.__setattr__(self, "theta_min", zero)
            object.__setattr__(self, "theta_max", zero)
            return
        if lo == hi:
            # A degenerate segment is a single point: normalize it as a Point,
            # so that equal points give equal segments.
            p = Point(self.u, self.v, lo)
            object.__setattr__(self, "u", p.u)
            object.__setattr__(self, "v", p.v)
            object.__setattr__(self, "theta_min", p.theta)
            object.__setattr__(self, "theta_max", p.theta)
            return
        u, v, swapped = orient(self.u, self.v)
        if swapped:
            lo, hi = (zero + 1) - hi, (zero + 1) - lo
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "theta_min", lo)
        object.__setattr__(self, "theta_max", hi)

    @property
    def is_point(self) -> bool:
        """Whether the segment is degenerate, i.e. a single point."""
        return self.theta_min == self.theta_max

    @property
    def length_fraction(self) -> float:
        """``theta_max - theta_min``; multiply by the edge length for a length."""
        return self.theta_max - self.theta_min

    def endpoints(self) -> Tuple[Point, Point]:
        """Return the two endpoints of the segment as :class:`Point` objects."""
        return (
            Point(self.u, self.v, self.theta_min),
            Point(self.u, self.v, self.theta_max),
        )

    def __iter__(self) -> Iterator[Point]:
        return iter(self.endpoints())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if self.u == self.v:
            return f"Segment({self.u!r})"
        if self.is_point:
            return f"Segment({self.u!r}, {self.v!r}, {self.theta_min!r})"
        return (
            f"Segment({self.u!r}, {self.v!r}, "
            f"{self.theta_min!r}, {self.theta_max!r})"
        )
