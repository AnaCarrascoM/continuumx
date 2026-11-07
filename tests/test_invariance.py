"""Subdivision invariance and scaling.

Subdividing an edge of length ``l`` into two edges of lengths ``t*l`` and
``(1-t)*l`` produces a graph whose continuous graph is *isometric* to the
original.  No continuous parameter may therefore change.  This is a strong
end-to-end test: it exercises the point representation, the distance, the hat
functions and the envelopes at once, and it is sensitive to exactly the kind of
error that a hand-checked example on a small graph would miss.
"""

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
