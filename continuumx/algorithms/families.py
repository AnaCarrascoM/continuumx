"""Closed forms and specialized algorithms for trees and cycles.

On these two families every parameter has a simple description
(Propositions 4.3 to 4.5), which realizes the two extremes of
rad ≤ diam ≤ 2 rad.  These functions are cheaper than the general
algorithms, and the test suite uses them as independent checks.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Hashable, List, Optional, Tuple

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


def _tree_distances(
    G, source: Hashable
) -> Tuple[Dict[Hashable, Any], Dict[Hashable, Optional[Hashable]]]:
    """Distances from ``source`` in a tree, and the parent of each vertex. O(n).

    In a tree the path between two vertices is unique, so a single traversal
    (here, breadth-first) already yields the distances by adding up lengths
    along it: no priority queue is needed.
    """
    ell = G.edge_length(*G.edges[0])
    # 0 with the numeric type of the lengths (exact for Fraction)
    zero = ell - ell
    dist: Dict[Hashable, Any] = {source: zero}
    parent: Dict[Hashable, Optional[Hashable]] = {source: None}
    queue = deque([source])
    while queue:
        a = queue.popleft()
        for b in G.graph.neighbors(a):
            if b not in dist:
                dist[b] = dist[a] + G.edge_length(a, b)
                parent[b] = a
                queue.append(b)
    return dist, parent


def _double_sweep(
    G,
) -> Tuple[Hashable, Hashable, Any, Dict[Hashable, Optional[Hashable]]]:
    """Return ``(u, v, d(u, v), parent)`` for a diametral pair of a tree.

    ``parent`` describes the traversal rooted at ``u``, so following it from
    ``v`` walks back along the diametral path.
    """
    d0, _ = _tree_distances(G, G.nodes[0])
    u = max(d0, key=d0.__getitem__)
    du, parent = _tree_distances(G, u)
    v = max(du, key=du.__getitem__)
    return u, v, du[v], parent


def tree_diameter(G, *, length: str = "length"):
    """Return diam(𝒯) for a continuous tree.

    By Proposition 4.4, every diametral pair of a continuous tree consists of
    two leaves, so the continuous diameter equals the discrete one.

    Complexity
    ----------
    O(n), with two traversals of the tree, against O(m²) for the general
    algorithm.

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
    """Return the center of a continuous tree: the midpoint of a diametral path.

    By Proposition 4.5, the center of a continuous tree is always a single
    point.

    Complexity
    ----------
    O(n): two traversals of the tree, and a walk along the diametral path.

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

    u, v, diam, parent = _double_sweep(G)
    path = [v]
    while path[-1] != u:
        path.append(parent[path[-1]])
    path.reverse()
    half = diam / 2
    # 0 with the numeric type of the lengths (exact for Fraction)
    walked = half - half
    for a, b in zip(path, path[1:]):
        ell = G.edge_length(a, b)
        if walked + ell >= half:
            return Point(a, b, (half - walked) / ell)
        walked += ell
    return Point.at_vertex(v)  # pragma: no cover - unreachable


def cycle_diameter(G, *, length: str = "length"):
    """Return diam(𝒞) = Per(𝒞)/2 (Proposition 4.3).

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
    """Return the center of a continuous cycle: the whole graph."""
    G = as_continuous(G, length=length)
    if not is_cycle(G):
        raise ValueError("cycle_center requires a cycle")
    # 0 with the numeric type of the lengths (exact for Fraction)
    zero = G.total_length() - G.total_length()
    return [Segment(u, v, zero, zero + 1) for u, v in G.edges]
