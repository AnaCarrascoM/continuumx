"""Shared fixtures and the brute-force oracle used across the test suite.

The oracle is deliberately *independent* of the library: it subdivides every
edge into ``k`` equal pieces and runs plain ``networkx`` shortest paths on the
resulting weighted graph.  Distances between sampled points are then exact
continuous distances, and the sampled diameter and radius bracket the true ones
with an error of at most ``max_edge_length / k``.
"""

from __future__ import annotations

import random
from typing import Dict, Hashable, List, Tuple

import networkx as nx
import pytest

import continuumx as cx

SAMPLES = 24


def subdivide(G: cx.ContinuousGraph, k: int = SAMPLES):
    """Return ``(graph, samples)``: the k-subdivision and the sampled points.

    ``samples`` maps a :class:`continuumx.Point` to the node representing it in
    the subdivided graph.
    """
    out = nx.Graph()
    samples: Dict[cx.Point, Hashable] = {}
    for w in G.nodes:
        out.add_node(("v", w))
        samples[cx.Point.at_vertex(w)] = ("v", w)
    for u, v, ell in G.iter_edges_with_length():
        step = ell / k
        previous = ("v", u)
        for i in range(1, k):
            node = ("e", u, v, i)
            out.add_edge(previous, node, length=step)
            samples[cx.Point(u, v, i / k)] = node
            previous = node
        out.add_edge(previous, ("v", v), length=step)
    return out, samples


def oracle_distances(G: cx.ContinuousGraph, k: int = SAMPLES):
    """Return ``(samples, dist)`` with exact distances between sampled points."""
    sub, samples = subdivide(G, k)
    dist = dict(nx.all_pairs_dijkstra_path_length(sub, weight="length"))
    return samples, dist


def oracle_diameter(G: cx.ContinuousGraph, k: int = SAMPLES) -> float:
    samples, dist = oracle_distances(G, k)
    nodes = list(samples.values())
    return max(dist[a][b] for a in nodes for b in nodes)


def oracle_radius(G: cx.ContinuousGraph, k: int = SAMPLES) -> float:
    samples, dist = oracle_distances(G, k)
    nodes = list(samples.values())
    return min(max(dist[a][b] for b in nodes) for a in nodes)


def sampling_error(G: cx.ContinuousGraph, k: int = SAMPLES) -> float:
    """Upper bound on the error committed by the oracle."""
    return max(ell for _, _, ell in G.iter_edges_with_length()) / k


def random_graph(seed: int, n: int = 7, p: float = 0.35,
                 weighted: bool = True) -> cx.ContinuousGraph:
    """Return a random connected continuous graph."""
    rng = random.Random(seed)
    while True:
        g = nx.gnp_random_graph(n, p, seed=rng.randint(0, 10 ** 6))
        if g.number_of_edges() and nx.is_connected(g):
            break
    for u, v in g.edges():
        g[u][v]["length"] = round(rng.uniform(0.5, 6.0), 3) if weighted else 1.0
    return cx.ContinuousGraph(g)


RANDOM_SEEDS = list(range(12))


def as_interval(x):
    """Return ``(u, v, theta_min, theta_max)`` for an element of a center.

    The center mixes :class:`continuumx.Point` (an isolated central point) and
    :class:`continuumx.Segment`; a point is read as a degenerate interval.
    """
    if isinstance(x, cx.Point):
        return x.u, x.v, x.theta, x.theta
    return x.u, x.v, x.theta_min, x.theta_max


@pytest.fixture
def triangle_1_1_5() -> cx.ContinuousGraph:
    """The triangle with lengths 1, 1, 5: a continuous graph that is not geometric.

    It is the smallest example separating the naive same-edge formula
    ``|lambda_p - lambda_q|`` from the true distance, and the naive same-edge
    diameter ``l(e)`` from the true one.
    """
    g = nx.Graph()
    g.add_edge("a", "b", length=5.0)
    g.add_edge("a", "c", length=1.0)
    g.add_edge("c", "b", length=1.0)
    return cx.ContinuousGraph(g)


@pytest.fixture
def emergency_network() -> cx.ContinuousGraph:
    """A weighted network whose continuous center is a union of three sub-paths."""
    g = nx.Graph()
    for u, v, ell in [
        ("a", "b", 2.0),
        ("b", "c", 3.0),
        ("c", "d", 4.0),
        ("d", "e", 2.0),
        ("b", "e", 6.0),
    ]:
        g.add_edge(u, v, length=ell)
    return cx.ContinuousGraph(g)


@pytest.fixture
def interior_center_network() -> cx.ContinuousGraph:
    """A weighted network whose continuous center is a single interior point.

    It contains the cycle ``b c d e b`` and a pendant edge ``ab``; the center
    lies on ``ab``, at distance 4.75 from ``a``, and no vertex attains the
    radius.  This is the situation the introduction illustrates with a rural
    road network.
    """
    g = nx.Graph()
    for u, v, ell in [
        ("a", "b", 5.0),
        ("b", "c", 1.0),
        ("b", "e", 1.0),
        ("c", "d", 5.0),
        ("d", "e", 2.0),
    ]:
        g.add_edge(u, v, length=ell)
    return cx.ContinuousGraph(g)
