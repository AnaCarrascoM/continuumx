"""Exact arithmetic: with ``Fraction`` lengths nothing may degrade to ``float``.

The library promises (see :class:`ContinuousGraph`) that Fraction inputs give
exact rational outputs.  A single stray ``0.0`` or ``1.0`` in an intermediate
step silently breaks that promise, because ``Fraction + float`` is a float, so
these tests check the *type* of every number returned, not only its value.
"""

from fractions import Fraction as F

import networkx as nx
import pytest

import continuumx as cx


def _graph(edges):
    g = nx.Graph()
    for u, v, ell in edges:
        g.add_edge(u, v, length=F(ell))
    return cx.ContinuousGraph(g, tol=0)


@pytest.fixture
def coding_example():
    # The running example of the coding chapter.
    return _graph([("a", "b", 2), ("b", "c", 3), ("b", "e", 6),
                   ("c", "d", 4), ("c", "f", 3), ("d", "e", 2)])


@pytest.fixture
def logistics():
    return _graph([("a", "b", 10), ("b", "c", 5), ("a", "c", 20),
                   ("a", "d", 5), ("d", "e", 15)])


def _is_exact(x):
    return isinstance(x, (F, int)) and not isinstance(x, bool)


def test_parameters_are_exact(coding_example):
    G = coding_example
    assert cx.diameter(G) == F(21, 2) and _is_exact(cx.diameter(G))
    assert cx.radius(G) == F(15, 2) and _is_exact(cx.radius(G))


def test_center_is_exact(coding_example):
    segs = cx.center(coding_example)
    assert [(s.u, s.v, s.theta_min, s.theta_max) for s in segs] == [
        ("b", "c", F(0), F(1)),
        ("b", "e", F(0), F(1, 4)),
        ("c", "d", F(0), F(5, 8)),
    ]
    for s in segs:
        assert _is_exact(s.theta_min) and _is_exact(s.theta_max)


def test_logistics_example_is_exact(logistics):
    G = logistics
    assert cx.diameter(G) == F(75, 2) and _is_exact(cx.diameter(G))
    assert cx.radius(G) == F(75, 4) and _is_exact(cx.radius(G))
    (seg,) = cx.center(G)
    assert (seg.u, seg.v) == ("a", "d")
    assert seg.theta_min == seg.theta_max == F(1, 4)
    assert _is_exact(seg.theta_min)


def test_diametral_pair_and_farthest_point_are_exact(logistics):
    p, q = cx.diametral_pair(logistics)
    assert _is_exact(p.theta) and _is_exact(q.theta)
    assert cx.distance(logistics, p, q) == cx.diameter(logistics)
    far = cx.farthest_point_on_edge(logistics, logistics.point("d"), "a", "c")
    assert _is_exact(far.theta)


def test_eccentricity_function_is_exact(logistics):
    f = cx.eccentricity_function(logistics, "a", "c")
    assert all(_is_exact(x) for x in f.xs)
    assert all(_is_exact(y) for y in f.ys)


def test_hat_functions_are_exact(logistics):
    for hat, _ in cx.hat_functions(logistics, "a", "c"):
        assert _is_exact(hat.b) and _is_exact(hat.s) and _is_exact(hat.h)
        assert all(_is_exact(x) for x in hat.breakpoints().xs)


def test_families_are_exact():
    tree = _graph([(0, 1, 1), (1, 2, 2), (2, 3, 4)])
    c = cx.tree_center(tree)
    assert _is_exact(c.theta) and cx.distance(tree, c, tree.point(0)) == F(7, 2)
    cyc = _graph([(0, 1, 1), (1, 2, 2), (2, 0, 4)])
    assert cx.cycle_diameter(cyc) == F(7, 2)
    for s in cx.cycle_center(cyc):
        assert _is_exact(s.theta_min) and _is_exact(s.theta_max)


def test_fraction_and_float_runs_agree():
    import random
    rng = random.Random(7)
    for _ in range(25):
        n = rng.randint(3, 7)
        while True:
            g = nx.gnp_random_graph(n, 0.6, seed=rng.randint(0, 10**6))
            if nx.is_connected(g):
                break
        lengths = {e: F(rng.randint(1, 12), rng.choice([1, 2, 4])) for e in g.edges}
        gf, gx = nx.Graph(), nx.Graph()
        for (u, v), ell in lengths.items():
            gf.add_edge(u, v, length=ell)
            gx.add_edge(u, v, length=float(ell))
        Gf, Gx = cx.ContinuousGraph(gf, tol=0), cx.ContinuousGraph(gx)
        assert float(cx.diameter(Gf)) == pytest.approx(cx.diameter(Gx))
        assert float(cx.radius(Gf)) == pytest.approx(cx.radius(Gx))
        assert _is_exact(cx.diameter(Gf)) and _is_exact(cx.radius(Gf))
        for s in cx.center(Gf):
            assert _is_exact(s.theta_min) and _is_exact(s.theta_max)
