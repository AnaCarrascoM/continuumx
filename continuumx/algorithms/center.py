"""Center of a continuous graph, and the local center of an edge.

Unlike its discrete counterpart, the continuous center

    C(𝒢) = { p ∈ 𝒢 : ecc(p) = rad(𝒢) }

need not be finite, and need not contain a vertex: on a cycle it is the whole
graph, and on the network of the introduction it is a single interior point of
an edge.  It is therefore returned as a list whose elements are either a
:class:`~continuumx.Point`, an isolated central point, or a
:class:`~continuumx.Segment`, a stretch of an edge made of central points.
"""

from __future__ import annotations

from typing import Hashable, List, Union

from continuumx.algorithms.eccentricity import eccentricity_function
from continuumx.algorithms.radius import radius as _radius
from continuumx.classes.continuous_graph import as_continuous
from continuumx.classes.point import Point, Segment, node_key

__all__ = ["center", "local_center"]

Element = Union[Point, Segment]


def _element(u: Hashable, v: Hashable, lo, hi) -> Element:
    """A point if the interval is degenerate, a segment otherwise."""
    if lo == hi:
        return Point(u, v, lo)  # a vertex is normalized by Point
    return Segment(u, v, lo, hi)


def local_center(
    G, u: Hashable, v: Hashable, *, length: str = "length"
) -> List[Element]:
    """Return the local center of the edge ``uv``.

    That is, the set of points of ``uv`` attaining
    :func:`~continuumx.local_radius`, as a list of points and segments.

    Examples
    --------
    >>> import networkx as nx, continuumx as cx
    >>> G = cx.ContinuousGraph(nx.path_graph(3))
    >>> cx.local_center(G, 0, 1)
    [Point(1)]
    """
    G = as_continuous(G, length=length)
    _, intervals = eccentricity_function(G, u, v).minimum(tol=G.tol)
    return [_element(u, v, lo, hi) for lo, hi in intervals]


def center(G, *, length: str = "length") -> List[Element]:
    """Return the continuous center C(𝒢).

    The center is the union, over the edges whose local radius equals
    rad(𝒢), of their local centers.  Each
    element of the list is a :class:`~continuumx.Point` when the minimum of an
    envelope is attained at an isolated point, and a
    :class:`~continuumx.Segment` when it is attained along a stretch of the
    edge.

    Complexity
    ----------
    O(n m log n + m³), the cost of
    :func:`~continuumx.radius`, since the envelopes are recomputed once.

    Examples
    --------
    >>> import networkx as nx, continuumx as cx
    >>> cx.center(cx.ContinuousGraph(nx.path_graph(3)))     # the middle vertex
    [Point(1)]
    >>> segs = cx.center(cx.ContinuousGraph(nx.cycle_graph(4)))
    >>> len(segs)                                           # the whole cycle
    4
    >>> all(s.length_fraction == 1.0 for s in segs)
    True
    """
    G = as_continuous(G, length=length)
    if G.number_of_edges() == 0:
        return [Point.at_vertex(G.nodes[0])]

    rad = _radius(G)
    tol = G.tol
    items: List[Element] = []
    for u, v in G.edges:
        value, intervals = eccentricity_function(G, u, v).minimum(tol=tol)
        if value > rad + tol:
            continue
        items.extend(_element(u, v, lo, hi) for lo, hi in intervals)

    return _tidy(items)


def _tidy(items: List[Element]) -> List[Element]:
    """Drop duplicates, and vertices already covered by a segment."""
    unique = list(dict.fromkeys(items))
    covered = {
        p
        for x in unique
        if isinstance(x, Segment)
        for p in x.endpoints()
        if p.is_vertex
    }
    kept = [x for x in unique if not (isinstance(x, Point) and x in covered)]
    return sorted(kept, key=_sort_key)


def _sort_key(x: Element):
    theta = x.theta if isinstance(x, Point) else x.theta_min
    return (node_key(x.u), node_key(x.v), theta)
