"""Diameter of a continuous graph.

The diameter is a maximum over an infinite set of pairs of points, yet it is
determined by finitely many candidates, given by the hat functions of
:mod:`continuumx.algorithms.eccentricity` (Corollaries 4.1 and 4.2):

    diam(𝒢) = max over p ∈ 𝒢 of ecc(p)
            = max over edges e, hats k, and θ ∈ [0, 1] of hat_k^e(θ)
            = max over edges e and hats k of h_k^e,

because the maximum of a hat function is its ceiling.  The diameter is
therefore the largest ceiling among the m(m+1) hats of the graph, with no case
analysis and no separate treatment of the pair of points lying on the same
edge — the degenerate hats of an edge against itself already encode it.
"""

from __future__ import annotations

from typing import Tuple

from continuumx.algorithms.eccentricity import farthest_point_on_edge, hat_functions
from continuumx.classes.continuous_graph import as_continuous
from continuumx.classes.point import Point

__all__ = ["diameter", "diametral_pair"]


def diameter(G, *, length: str = "length"):
    """Return the continuous diameter diam(𝒢).

    Complexity
    ----------
    O(n m log n) for the all-pairs shortest paths plus O(m²) for the ceilings.

    Notes
    -----
    The largest distance between two points of the same edge uv is not ℓ(uv)
    but (ℓ(uv) + d_G(u, v)) / 2, by (4.16): on the triangle with ℓ(ab) = 5 and
    a path a c b of length 2, it is 3.5, not 5.

    Examples
    --------
    >>> import networkx as nx, continuumx as cx
    >>> cx.diameter(cx.ContinuousGraph(nx.cycle_graph(5)))   # perimeter / 2
    2.5
    >>> cx.diameter(cx.ContinuousGraph(nx.path_graph(4)))    # a tree: discrete
    3.0
    """
    G = as_continuous(G, length=length)
    if G.number_of_edges() == 0:
        return 0.0
    G.precompute()
    return max(hat.h for u, v in G.edges for hat, _ in hat_functions(G, u, v))


def diametral_pair(G, *, length: str = "length") -> Tuple[Point, Point]:
    """Return one diametral pair (p, q), i.e. with ``distance(p, q) == diameter``.

    In general there are infinitely many; this returns one witness, as in
    Step 3 of the diameter algorithm of Chen and Garfinkel.  Let uv be the edge
    and wz the witness edge of the highest hat.  The first point x* is the
    point of uv of parameter θ* = b_wz, the left end of the interval
    [b_wz, s_wz] where the hat attains its ceiling; the second is the point of
    wz farthest from x*.
    For n ≥ 3, the pairs (uv, uv) are omitted, since for every edge uv there
    is a distinct edge vw with d_max(uv, uv) ≤ d_max(uv, vw); this also keeps
    :func:`farthest_point_on_edge` within the hypotheses of the lemma it
    implements (x* not interior to wz).  Returns a pair of copies of the only
    vertex when the graph has no edges.

    Examples
    --------
    >>> import networkx as nx, continuumx as cx
    >>> G = cx.ContinuousGraph(nx.cycle_graph(4))
    >>> p, q = cx.diametral_pair(G)
    >>> round(cx.distance(G, p, q), 10) == cx.diameter(G)
    True
    """
    G = as_continuous(G, length=length)
    if G.number_of_edges() == 0:
        w = Point.at_vertex(G.nodes[0])
        return (w, w)
    G.precompute()

    skip_same_edge = G.number_of_nodes() >= 3
    best = None
    for u, v in G.edges:
        for hat, witness in hat_functions(G, u, v):
            if skip_same_edge and set(witness) == {u, v}:
                continue  # pairs (uv, uv) are omitted
            if best is None or hat.h > best[0]:
                best = (hat.h, u, v, hat, witness)

    assert best is not None  # the graph has at least one edge
    _, u, v, hat, witness = best
    p = Point(u, v, hat.b)  # theta* = b_wz
    q = farthest_point_on_edge(G, p, witness[0], witness[1])
    return (p, q)
