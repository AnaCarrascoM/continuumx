"""Subdivision invariance, exact arithmetic and scaling.

Subdividing an edge of length ``l`` into two edges of lengths ``t*l`` and
``(1-t)*l`` produces a graph whose continuous graph is *isometric* to the
original.  No continuous parameter may therefore change.  This is a strong
end-to-end test: it exercises the point representation, the distance, the hat
functions and the envelopes at once, and it is sensitive to exactly the kind of
error that a hand-checked example on a small graph would miss.
"""

from fractions import Fraction

import networkx as nx
import pytest

import continuumx as cx
from conftest import RANDOM_SEEDS, random_graph


def subdivide_once(G: cx.ContinuousGraph, u, v, t: float = 0.4):
    """Return a copy of ``G`` with the edge ``uv`` split at parameter ``t``."""
    g = G.graph.copy()
    ell = G.edge_length(u, v)
    mid = ("mid", u, v)
    g.remove_edge(u, v)
    g.add_edge(u, mid, length=t * ell)
    g.add_edge(mid, v, length=(1 - t) * ell)
    return cx.ContinuousGraph(g, length=G.length_attr)


@pytest.mark.parametrize("seed", RANDOM_SEEDS[:8])
def test_parameters_are_invariant_under_subdivision(seed):
    G = random_graph(seed)
    u, v = G.edges[0]
    H = subdivide_once(G, u, v)
    assert cx.diameter(H) == pytest.approx(cx.diameter(G), abs=1e-9)
    assert cx.radius(H) == pytest.approx(cx.radius(G), abs=1e-9)


@pytest.mark.parametrize("seed", RANDOM_SEEDS[:6])
def test_parameters_scale_linearly_with_the_lengths(seed):
    G = random_graph(seed)
    g = G.graph.copy()
    for a, b in g.edges():
        g[a][b]["length"] *= 3.0
    H = cx.ContinuousGraph(g)
    assert cx.diameter(H) == pytest.approx(3 * cx.diameter(G))
    assert cx.radius(H) == pytest.approx(3 * cx.radius(G))


def test_exact_arithmetic_with_fractions():
    """With rational lengths every parameter is an exact rational."""
    g = nx.Graph()
    g.add_edge("a", "b", length=Fraction(5))
    g.add_edge("a", "c", length=Fraction(1))
    g.add_edge("c", "b", length=Fraction(1))
    G = cx.ContinuousGraph(g, tol=0)

    diam = cx.diameter(G)
    rad = cx.radius(G)
    assert isinstance(diam, Fraction) and diam == Fraction(7, 2)
    assert isinstance(rad, Fraction) and rad == Fraction(7, 2)

    p = cx.Point("a", "b", Fraction(1, 50))
    q = cx.Point("a", "b", Fraction(49, 50))
    assert cx.distance(G, p, q) == Fraction(11, 5)


def test_exact_center_of_a_path():
    g = nx.path_graph(4)
    for u, v in g.edges():
        g[u][v]["length"] = Fraction(1)
    G = cx.ContinuousGraph(g, tol=0)
    (c,) = cx.center(G)
    assert isinstance(c, cx.Point) and c.theta == Fraction(1, 2)
    assert cx.radius(G) == Fraction(3, 2)
