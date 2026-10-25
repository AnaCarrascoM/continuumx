"""Closed forms and specialised algorithms for trees and cycles.

Two families admit an exact description of every metric parameter, and they
realise the two extremes of rad ≤ diam ≤ 2 rad:

* on a **tree**, the continuous diameter coincides with the discrete one, and
  rad(𝒯) = diam(𝒯)/2, the centre being the midpoint of any diametral path (an
  interior point of an edge in general);
* on a **cycle**, every point has the same eccentricity, so
  rad(𝒞) = diam(𝒞) = Per(𝒞)/2 and the centre is the whole graph.

These functions are cheaper than the general algorithms, and they double as
independent oracles for the test suite.
"""

from __future__ import annotations

from typing import Hashable, List, Tuple

import networkx as nx

from continuumx.classes.continuous_graph import as_continuous
from continuumx.classes.point import Point, Segment

__all__ = [
    "is_tree",
    "is_cycle",
    "tree_diameter",
    "tree_radius",
    "tree_center",
    "cycle_diameter",
    "cycle_radius",
    "cycle_center",
]


def is_tree(G, *, length: str = "length") -> bool:
    """Whether the underlying combinatorial graph is a tree."""
    G = as_continuous(G, length=length)
    return G.number_of_edges() == G.number_of_nodes() - 1


def is_cycle(G, *, length: str = "length") -> bool:
    """Whether the underlying combinatorial graph is a cycle."""
    G = as_continuous(G, length=length)
    if G.number_of_nodes() < 3:
        return False
    return all(deg == 2 for _, deg in G.graph.degree())


def _double_sweep(G) -> Tuple[Hashable, Hashable, float]:
    """Return ``(u, v, d(u, v))`` for a diametral pair of vertices of a tree."""
    start = G.nodes[0]
    d0 = G.distances_from(start)
    u = max(d0, key=d0.__getitem__)
    du = G.distances_from(u)
    v = max(du, key=du.__getitem__)
    return u, v, du[v]


def tree_diameter(G, *, length: str = "length"):
    """Return diam(𝒯) for a continuous tree.

    On a tree the longest path joins two leaves, and no pair involving an
    interior point of an edge can beat it, so the continuous diameter equals
    the discrete one.  It is found by two shortest-path traversals.

    Complexity
    ----------
    O(m log n) — two Dijkstra runs — against O(m²) for the general algorithm.

    Examples
    --------
    >>> import networkx as nx, continuumx as cx
    >>> G = cx.ContinuousGraph(nx.path_graph(4))
    >>> cx.tree_diameter(G) == cx.diameter(G) == 3.0
    True
    """
    G = as_continuous(G, length=length)
    if not is_tree(G):
        raise ValueError("tree_diameter requires a tree")
    if G.number_of_edges() == 0:
        return 0.0
    return _double_sweep(G)[2]


def tree_radius(G, *, length: str = "length"):
    """Return rad(𝒯) = diam(𝒯)/2."""
    return tree_diameter(G, length=length) / 2


def tree_center(G, *, length: str = "length") -> Point:
    """Return the continuous centre of a tree: the midpoint of a diametral path.

    In the continuous setting the centre of a tree is always a **single**
    point; the parity obstruction of the discrete case, where an odd diameter
    forces two central vertices, disappears.

    Examples
    --------
    >>> import networkx as nx, continuumx as cx
    >>> G = cx.ContinuousGraph(nx.path_graph(3))     # even diameter: a vertex
    >>> cx.tree_center(G)
    Point(1)
    >>> H = cx.ContinuousGraph(nx.path_graph(4))     # odd: inside an edge
    >>> cx.tree_center(H)
    Point(1, 2, 0.5)
    """
    G = as_continuous(G, length=length)
    if not is_tree(G):
        raise ValueError("tree_center requires a tree")
    if G.number_of_edges() == 0:
        return Point.at_vertex(G.nodes[0])

    u, v, diam = _double_sweep(G)
    path = nx.shortest_path(G.graph, u, v, weight=G.length_attr)
    half = diam / 2
    walked = half - half   # same numeric type as the lengths (exact for Fraction)
    for a, b in zip(path, path[1:]):
        ell = G.edge_length(a, b)
        if walked + ell >= half:
            return Point(a, b, (half - walked) / ell)
        walked += ell
    return Point.at_vertex(v)  # pragma: no cover - unreachable


def cycle_diameter(G, *, length: str = "length"):
    """Return diam(𝒞) = Per(𝒞)/2.

    Examples
    --------
    >>> import networkx as nx, continuumx as cx
    >>> G = cx.ContinuousGraph(nx.cycle_graph(5))
    >>> cx.cycle_diameter(G) == cx.diameter(G) == 2.5
    True
    """
    G = as_continuous(G, length=length)
    if not is_cycle(G):
        raise ValueError("cycle_diameter requires a cycle")
    return G.total_length() / 2


def cycle_radius(G, *, length: str = "length"):
    """Return rad(𝒞) = diam(𝒞)."""
    return cycle_diameter(G, length=length)


def cycle_center(G, *, length: str = "length") -> List[Segment]:
    """Return the continuous centre of a cycle: the whole graph."""
    G = as_continuous(G, length=length)
    if not is_cycle(G):
        raise ValueError("cycle_center requires a cycle")
    zero = G.total_length() - G.total_length()   # numeric type of the lengths
    return [Segment(u, v, zero, zero + 1) for u, v in G.edges]
