"""Distances between points of a continuous graph.

The distance d_𝒢 is the length of a shortest walk between two points
(Definition 4.3).  Unless both points lie on the same edge, a shortest walk
leaves the edge of the first point through one of its endpoints and enters the
edge of the second through one of its endpoints.  This reduces the distance to
a minimum of finitely many expressions in the distances d_G between vertices
(Proposition 4.1).
"""

from __future__ import annotations

from typing import Hashable

from continuumx.classes.continuous_graph import as_continuous
from continuumx.classes.point import Point

__all__ = ["distance", "distance_to_vertex"]


def distance_to_vertex(G, p: Point, w: Hashable, *, length: str = "length"):
    """Return d_𝒢(p, w) for a point ``p`` and a vertex ``w``.

    A shortest path leaves the edge uv containing ``p`` through ``u`` or
    through ``v``, so, with θ the parameter of ``p`` (formula (4.2)),

        d_𝒢(p, w) = min( θ ℓ(uv) + d_G(u, w),  (1 - θ) ℓ(uv) + d_G(v, w) ).
    """
    G = as_continuous(G, length=length)
    G.check_point(p)
    if p.is_vertex:
        return G.vertex_distance(p.u, w)
    u, v, theta = p.u, p.v, p.theta
    l_uv = G.edge_length(u, v)
    return min(theta * l_uv + G.vertex_distance(u, w),
               (1 - theta) * l_uv + G.vertex_distance(v, w))


def distance(G, p: Point, q: Point, *, length: str = "length"):
    """Return the continuous distance d_𝒢(p, q).

    Parameters
    ----------
    G : ContinuousGraph or networkx.Graph
    p, q : Point
    length : str, optional
        Edge-length attribute, used only if ``G`` still has to be wrapped.

    Notes
    -----
    Write θ for the parameter of ``p`` on uv and μ for that of ``q`` on wz.
    Three cases arise, those of Proposition 4.1.

    *One of the points is a vertex.*  Handled by :func:`distance_to_vertex`.

    *The points lie on different edges* uv and wz.  A path leaves uv through
    ``u`` or ``v`` and enters wz through ``w`` or ``z``, giving the minimum of
    the four expressions of (4.5).

    *The points lie on the same edge* uv.  It is **not** enough to take
    (μ - θ) ℓ(uv): a path may leave the edge through one endpoint and come
    back through the other, which is shorter whenever the edge is long
    compared with the rest of the graph.  For example, on the triangle with
    ℓ(ab) = 5 and a path a c b of total length 2, the points of ab at distance
    0.1 from a and 0.1 from b are at distance 2.2, not 4.8.  Ordering the
    parameters so that θ ≤ μ, the correct expression is (4.4):

        d_𝒢(p, q) = min( (μ - θ) ℓ(uv),
                         θ ℓ(uv) + d_G(u, v) + (1 - μ) ℓ(uv) ).

    Using d_G(u, v) rather than the distance in G - uv is safe: each expression
    is the length of an actual walk, so none can undercut the true distance,
    and the shortest path is among them.

    Complexity
    ----------
    O(1) once the shortest-path distances from the four endpoints involved are
    cached; each uncached source costs one Dijkstra run.

    Examples
    --------
    >>> import networkx as nx, continuumx as cx
    >>> g = nx.Graph()
    >>> g.add_edge("a", "b", length=5.0)
    >>> g.add_edge("a", "c", length=1.0)
    >>> g.add_edge("c", "b", length=1.0)
    >>> G = cx.ContinuousGraph(g)
    >>> p = G.point("a", "b", offset=0.1)
    >>> q = G.point("a", "b", offset=4.9)
    >>> round(cx.distance(G, p, q), 10)
    2.2
    """
    G = as_continuous(G, length=length)
    G.check_point(p)
    G.check_point(q)

    if q.is_vertex:
        return distance_to_vertex(G, p, q.u)
    if p.is_vertex:
        return distance_to_vertex(G, q, p.u)

    (u, v), (w, z) = (p.u, p.v), (q.u, q.v)
    l_uv, l_wz = G.edge_length(u, v), G.edge_length(w, z)
    theta, mu = p.theta, q.theta
    d = G.vertex_distance                       # d_G, by Dijkstra

    if (u, v) == (w, z):
        # Same edge (4.4): stay inside, or leave through the endpoint nearer
        # to one point and come back through the other endpoint.
        theta, mu = min(theta, mu), max(theta, mu)
        return min((mu - theta) * l_uv,
                   theta * l_uv + d(u, v) + (1 - mu) * l_uv)

    return min(theta * l_uv + d(u, w) + mu * l_wz,          # different edges (4.5)
               theta * l_uv + d(u, z) + (1 - mu) * l_wz,
               (1 - theta) * l_uv + d(v, w) + mu * l_wz,
               (1 - theta) * l_uv + d(v, z) + (1 - mu) * l_wz)
