"""Distances between points of a continuous graph.

The continuous distance d_𝒢 is the geodesic distance of the metric space 𝒢.
Any path between two points must leave the edge containing the first through
one of its endpoints and enter the edge containing the second through one of
*its* endpoints, which reduces the distance to a minimum of finitely many
expressions involving only the discrete distances d_G between vertices.
"""

from __future__ import annotations

from typing import Hashable

from continuumx.classes.continuous_graph import as_continuous
from continuumx.classes.point import Point

__all__ = ["distance", "distance_to_vertex"]


def distance_to_vertex(G, p: Point, w: Hashable, *, length: str = "length"):
    """Return d_𝒢(p, w) for a point ``p`` and a vertex ``w``.

    A shortest path leaves the edge uv containing ``p`` through ``u`` or
    through ``v``, so

        d_𝒢(p, w) = min( λ + d_G(u, w),  (ℓ(uv) - λ) + d_G(v, w) ).
    """
    G = as_continuous(G, length=length)
    G.check_point(p)
    if p.is_vertex:
        return G.vertex_distance(p.u, w)
    u, v = p.u, p.v
    ell = G.edge_length(u, v)
    lam = p.theta * ell
    return min(lam + G.vertex_distance(u, w), (ell - lam) + G.vertex_distance(v, w))


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
    Three cases arise.

    *One of the points is a vertex.*  Handled by :func:`distance_to_vertex`.

    *The points lie on different edges* uv and rs.  A path leaves uv through
    ``u`` or ``v`` and enters rs through ``r`` or ``s``, giving the minimum of
    four expressions.

    *The points lie on the same edge* uv.  It is **not** enough to take
    |λ_p - λ_q|: a path may leave the edge through one endpoint and come back
    through the other, which is shorter whenever the edge is long compared with
    the rest of the graph.  For example, on the triangle with ℓ(ab) = 5 and a
    path a c b of total length 2, the points of ab at distance 0.1 from a and
    0.1 from b are at distance 2.2, not 4.8.  The correct expression is

        d_𝒢(p, q) = min( |λ_p - λ_q|,
                         λ_p + d_G(u, v) + (ℓ - λ_q),
                         (ℓ - λ_p) + d_G(u, v) + λ_q ).

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

    u, v = p.u, p.v
    r, s = q.u, q.v
    ell_p = G.edge_length(u, v)
    ell_q = G.edge_length(r, s)
    lam_p = p.theta * ell_p
    lam_q = q.theta * ell_q

    if (u, v) == (r, s):
        # Same edge: stay inside, or leave through one endpoint and return
        # through the other.
        d_uv = G.vertex_distance(u, v)
        return min(
            abs(lam_p - lam_q),
            lam_p + d_uv + (ell_q - lam_q),
            (ell_p - lam_p) + d_uv + lam_q,
        )

    return min(
        lam_p + G.vertex_distance(u, r) + lam_q,
        lam_p + G.vertex_distance(u, s) + (ell_q - lam_q),
        (ell_p - lam_p) + G.vertex_distance(v, r) + lam_q,
        (ell_p - lam_p) + G.vertex_distance(v, s) + (ell_q - lam_q),
    )
