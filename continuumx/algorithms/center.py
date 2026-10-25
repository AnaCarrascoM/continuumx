"""Centre of a continuous graph, and the local centre of an edge.

Unlike its discrete counterpart, the continuous centre

    C(𝒢) = { p ∈ 𝒢 : ecc(p) = rad(𝒢) }

need not be finite, and need not contain a vertex: on a cycle it is the whole
graph, and on the network of the introduction it is a single interior point of
an edge.  It is therefore returned as a list of
:class:`~continuumx.Segment` objects — a union of sub-paths, with single points
represented as degenerate segments.
"""

from __future__ import annotations

from typing import Hashable, List

from continuumx.algorithms.eccentricity import eccentricity_function
from continuumx.algorithms.radius import radius as _radius
from continuumx.classes.continuous_graph import as_continuous
from continuumx.classes.point import Point, Segment, node_key

__all__ = ["center", "local_center"]


def local_center(G, u: Hashable, v: Hashable, *,
                 length: str = "length") -> List[Segment]:
    """Return the local continuous centre of the edge ``uv``.

    That is, the set of points of ``uv`` attaining
    :func:`~continuumx.local_radius`, as a list of segments.

    Examples
    --------
    >>> import networkx as nx, continuumx as cx
    >>> G = cx.ContinuousGraph(nx.path_graph(3))
    >>> cx.local_center(G, 0, 1)
    [Segment(1)]
    """
    G = as_continuous(G, length=length)
    _, intervals = eccentricity_function(G, u, v).minimum(tol=G.tol)
    return [Segment(u, v, lo, hi) for lo, hi in intervals]


def center(G, *, length: str = "length") -> List[Segment]:
    """Return the continuous centre C(𝒢).

    The centre is the union, over the edges whose local continuous radius
    equals the continuous radius, of their local continuous centres.

    Complexity
    ----------
    O(n m log n + m² log m), the cost of
    :func:`~continuumx.radius`, since the envelopes are recomputed once.

    Examples
    --------
    >>> import networkx as nx, continuumx as cx
    >>> cx.center(cx.ContinuousGraph(nx.path_graph(3)))     # the middle vertex
    [Segment(1)]
    >>> segs = cx.center(cx.ContinuousGraph(nx.cycle_graph(4)))
    >>> len(segs)                                           # the whole cycle
    4
    >>> all(s.length_fraction == 1.0 for s in segs)
    True
    """
    G = as_continuous(G, length=length)
    if G.number_of_edges() == 0:
        return [Segment(G.nodes[0], G.nodes[0], 0.0, 0.0)]

    rad = _radius(G)
    tol = G.tol
    segments: List[Segment] = []
    for u, v in G.edges:
        value, intervals = eccentricity_function(G, u, v).minimum(tol=tol)
        if value > rad + tol:
            continue
        segments.extend(Segment(u, v, lo, hi) for lo, hi in intervals)

    return _tidy(segments)


def _tidy(segments: List[Segment]) -> List[Segment]:
    """Drop duplicates, and vertices already covered by a larger segment."""
    unique = list(dict.fromkeys(segments))
    covered = set()
    for seg in unique:
        if not seg.is_point:
            for endpoint in seg.endpoints():
                if endpoint.is_vertex:
                    covered.add(endpoint)
    out = [
        seg
        for seg in unique
        if not (seg.is_point and Point(seg.u, seg.v, seg.theta_min) in covered
                and seg.length_fraction == 0 and seg.u == seg.v)
    ]
    return sorted(out, key=lambda s: (node_key(s.u), node_key(s.v), s.theta_min))
