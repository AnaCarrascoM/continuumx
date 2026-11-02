"""Radius of a continuous graph, and the local radius of an edge.

Following Hansen, Labbé and Nicolas, the global problem is decomposed into one
local problem per edge.  The *local continuous radius* of an edge is the
smallest eccentricity attained on it,

    rad_c(uv) = min { ecc(x) : x ∈ uv },

and the continuous radius of the graph is the smallest of them:

    rad(𝒢) = min { rad_c(uv) : uv ∈ E }.

Each local problem is the minimization of the upper envelope of the m+1 hat
functions of the edge, a piecewise-linear function on [0, 1], computed in
O(m²) (see :func:`~continuumx.upper_envelope`).
"""

from __future__ import annotations

from typing import Hashable

from continuumx.algorithms.eccentricity import eccentricity_function
from continuumx.classes.continuous_graph import as_continuous

__all__ = ["radius", "local_radius"]


def local_radius(G, u: Hashable, v: Hashable, *, length: str = "length"):
    """Return the local continuous radius rad_c(uv).

    Complexity
    ----------
    O(m²) once the discrete distances are available.

    Examples
    --------
    >>> import networkx as nx, continuumx as cx
    >>> G = cx.ContinuousGraph(nx.path_graph(3))     # 0 - 1 - 2
    >>> cx.local_radius(G, 0, 1)                     # best point of edge 01 is 1
    1.0
    """
    G = as_continuous(G, length=length)
    value, _ = eccentricity_function(G, u, v).minimum(tol=G.tol)
    return value


def radius(G, *, length: str = "length"):
    """Return the continuous radius rad(𝒢).

    Complexity
    ----------
    O(n m log n) for the all-pairs shortest paths plus O(m³) for the m
    local problems.

    Examples
    --------
    >>> import networkx as nx, continuumx as cx
    >>> cx.radius(cx.ContinuousGraph(nx.path_graph(4)))   # a tree: diam / 2
    1.5
    >>> cx.radius(cx.ContinuousGraph(nx.cycle_graph(5)))  # a cycle: == diameter
    2.5
    """
    G = as_continuous(G, length=length)
    if G.number_of_edges() == 0:
        return 0.0
    G.precompute()
    return min(local_radius(G, u, v) for u, v in G.edges)
