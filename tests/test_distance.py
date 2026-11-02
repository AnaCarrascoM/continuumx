"""The continuous distance, against the subdivision oracle and the metric axioms."""

import itertools

import networkx as nx
import pytest

import continuumx as cx
from conftest import RANDOM_SEEDS, oracle_distances, random_graph


def test_same_edge_shortcut_is_not_the_absolute_difference(triangle_1_1_5):
    G = triangle_1_1_5
    p = G.point("a", "b", offset=0.1)
    q = G.point("a", "b", offset=4.9)
    # Naive answer: |4.9 - 0.1| = 4.8.  True answer: leave through a, come back
    # through b along the path of length 2.
    assert cx.distance(G, p, q) == pytest.approx(2.2)


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_distance_matches_the_subdivision_oracle(seed):
    G = random_graph(seed)
    samples, dist = oracle_distances(G, k=8)
    for p, q in itertools.combinations(list(samples)[:40], 2):
        expected = dist[samples[p]][samples[q]]
        assert cx.distance(G, p, q) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("seed", RANDOM_SEEDS[:6])
def test_metric_axioms(seed):
    G = random_graph(seed)
    points = [cx.Point(u, v, t) for u, v in G.edges for t in (0.0, 0.2, 0.5, 0.9)]
    for p in points:
        assert cx.distance(G, p, p) == pytest.approx(0.0)
    for p, q in itertools.combinations(points, 2):
        d = cx.distance(G, p, q)
        assert d > 0 or G.same_point(p, q)
        assert d == pytest.approx(cx.distance(G, q, p))
    for p, q, r in itertools.combinations(points[:14], 3):
        assert (
            cx.distance(G, p, r) <= cx.distance(G, p, q) + cx.distance(G, q, r) + 1e-9
        )


@pytest.mark.parametrize("seed", RANDOM_SEEDS[:6])
def test_distance_restricted_to_vertices_is_the_discrete_distance(seed):
    G = random_graph(seed)
    for u in G.nodes:
        for v in G.nodes:
            assert cx.distance(
                G, cx.Point.at_vertex(u), cx.Point.at_vertex(v)
            ) == pytest.approx(G.vertex_distance(u, v))


def test_distance_to_vertex_agrees_with_distance(triangle_1_1_5):
    G = triangle_1_1_5
    p = G.point("a", "b", theta=0.3)
    assert cx.distance_to_vertex(G, p, "c") == pytest.approx(
        cx.distance(G, p, G.point("c"))
    )


def test_distance_accepts_a_plain_networkx_graph():
    g = nx.path_graph(3)
    assert cx.distance(g, cx.Point(0, 1, 0.5), cx.Point(1, 2, 0.5)) == pytest.approx(
        1.0
    )
