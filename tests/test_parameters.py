"""Eccentricity, diameter, radius and center.

Cross-checked three ways: against the subdivision oracle, against the closed
forms for trees and cycles, and against the structural relations that hold in
any metric space.
"""

import networkx as nx
import pytest

import continuumx as cx
from conftest import (
    RANDOM_SEEDS,
    as_interval,
    oracle_diameter,
    oracle_distances,
    oracle_radius,
    random_graph,
    sampling_error,
)

GRID = [0.0, 0.13, 0.5, 0.77, 1.0]


# ----------------------------------------------------------------------
# Eccentricity
# ----------------------------------------------------------------------

@pytest.mark.parametrize("seed", RANDOM_SEEDS[:8])
def test_eccentricity_matches_the_oracle(seed):
    G = random_graph(seed)
    samples, dist = oracle_distances(G, k=16)
    err = sampling_error(G, 16)
    for p in list(samples)[:25]:
        approx = max(dist[samples[p]][target] for target in samples.values())
        exact = cx.eccentricity(G, p)
        assert approx - 1e-9 <= exact <= approx + err + 1e-9


@pytest.mark.parametrize("seed", RANDOM_SEEDS[:8])
def test_eccentricity_function_agrees_pointwise(seed):
    G = random_graph(seed)
    for u, v in G.edges:
        f = cx.eccentricity_function(G, u, v)
        for theta in GRID:
            assert f(theta) == pytest.approx(
                cx.eccentricity(G, cx.Point(u, v, theta)), abs=1e-9
            )


@pytest.mark.parametrize("seed", RANDOM_SEEDS[:6])
def test_farthest_point_on_edge_attains_the_remoteness(seed):
    G = random_graph(seed)
    for p in [cx.Point(u, v, 0.37) for u, v in G.edges]:
        for u, v in G.edges:
            q = cx.farthest_point_on_edge(G, p, u, v)
            assert cx.distance(G, p, q) == pytest.approx(
                cx.eccentricity_to_edge(G, p, u, v), abs=1e-9
            )


def test_eccentricity_of_a_vertex_can_exceed_the_discrete_one():
    # On a cycle of odd length the farthest point from a vertex is interior.
    g = nx.cycle_graph(5)
    G = cx.ContinuousGraph(g)
    assert nx.eccentricity(g, 0) == 2
    assert cx.eccentricity(G, G.point(0)) == pytest.approx(2.5)


# ----------------------------------------------------------------------
# Diameter
# ----------------------------------------------------------------------

@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_diameter_matches_the_oracle(seed):
    G = random_graph(seed)
    approx = oracle_diameter(G, k=20)
    err = sampling_error(G, 20)
    exact = cx.diameter(G)
    assert approx - 1e-9 <= exact <= approx + err + 1e-9


def test_diameter_does_not_overestimate_a_long_edge(triangle_1_1_5):
    # The naive "the two ends of the longest edge" answer would be 5.0.
    assert cx.diameter(triangle_1_1_5) == pytest.approx(3.5)


@pytest.mark.parametrize("seed", RANDOM_SEEDS[:8])
def test_diametral_pair_realizes_the_diameter(seed):
    G = random_graph(seed)
    p, q = cx.diametral_pair(G)
    assert cx.distance(G, p, q) == pytest.approx(cx.diameter(G), abs=1e-9)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 7])
def test_tree_diameter_is_the_discrete_one(n):
    g = nx.path_graph(n)
    G = cx.ContinuousGraph(g)
    assert cx.diameter(G) == pytest.approx(nx.diameter(g))
    assert cx.tree_diameter(G) == pytest.approx(cx.diameter(G))


@pytest.mark.parametrize("n", [3, 4, 5, 6, 9])
def test_cycle_diameter_is_half_the_perimeter(n):
    G = cx.ContinuousGraph(nx.cycle_graph(n))
    assert cx.diameter(G) == pytest.approx(n / 2)
    assert cx.cycle_diameter(G) == pytest.approx(n / 2)


# ----------------------------------------------------------------------
# Radius and center
# ----------------------------------------------------------------------

@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_radius_matches_the_oracle(seed):
    G = random_graph(seed)
    # The oracle only sees sampled points, so it may over- or under-shoot the
    # radius; both errors are bounded by half the sampling step.
    approx = oracle_radius(G, k=20)
    err = sampling_error(G, 20)
    assert abs(cx.radius(G) - approx) <= err + 1e-9


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_radius_diameter_inequalities(seed):
    G = random_graph(seed)
    rad, diam = cx.radius(G), cx.diameter(G)
    assert rad <= diam + 1e-9
    assert diam <= 2 * rad + 1e-9


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_center_points_have_eccentricity_equal_to_the_radius(seed):
    G = random_graph(seed)
    rad = cx.radius(G)
    center = cx.center(G)
    assert center
    for x in center:
        u, v, lo, hi = as_interval(x)
        for theta in (lo, (lo + hi) / 2, hi):
            p = cx.Point(u, v, theta)
            assert cx.eccentricity(G, p) == pytest.approx(rad, abs=1e-9)


@pytest.mark.parametrize("seed", RANDOM_SEEDS[:8])
def test_no_point_beats_the_radius(seed):
    G = random_graph(seed)
    rad = cx.radius(G)
    for u, v in G.edges:
        for theta in GRID:
            assert cx.eccentricity(G, cx.Point(u, v, theta)) >= rad - 1e-9


@pytest.mark.parametrize("n", [2, 3, 4, 5, 8])
def test_tree_radius_and_center(n):
    G = cx.ContinuousGraph(nx.path_graph(n))
    assert cx.radius(G) == pytest.approx(cx.diameter(G) / 2)
    assert cx.tree_radius(G) == pytest.approx(cx.radius(G))
    center = cx.center(G)
    assert len(center) == 1 and isinstance(center[0], cx.Point)
    assert center[0] == cx.tree_center(G)


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_cycle_radius_and_center(n):
    G = cx.ContinuousGraph(nx.cycle_graph(n))
    assert cx.radius(G) == pytest.approx(cx.diameter(G))
    segments = cx.center(G)
    assert len(segments) == n
    assert all(seg.length_fraction == pytest.approx(1.0) for seg in segments)


def test_center_may_be_interior_to_an_edge(interior_center_network):
    """Unlike the discrete center, the continuous center need not meet V."""
    G = interior_center_network
    (c,) = cx.center(G)
    assert isinstance(c, cx.Point) and not c.is_vertex
    assert G.offset(c) == pytest.approx(4.75)
    assert cx.eccentricity(G, c) == pytest.approx(cx.radius(G))
    assert all(
        cx.eccentricity(G, cx.Point.at_vertex(w)) > cx.radius(G) + 1e-9
        for w in G.nodes
    )


def test_center_may_be_a_union_of_subpaths(emergency_network):
    """It need not be finite either: here it is three sub-paths."""
    G = emergency_network
    segments = cx.center(G)
    assert len(segments) == 3
    assert any(not seg.is_point for seg in segments)
    rad = cx.radius(G)
    # Nothing outside the reported segments attains the radius.
    for u, v in G.edges:
        inside = [s for s in segments if (s.u, s.v) == (u, v)]
        for i in range(41):
            theta = i / 40
            p = cx.Point(u, v, theta)
            attains = abs(cx.eccentricity(G, p) - rad) <= 1e-9
            reported = any(
                s.theta_min - 1e-9 <= theta <= s.theta_max + 1e-9 for s in inside
            ) or (p.is_vertex and any(p in s.endpoints() for s in segments))
            assert attains == reported


def test_local_radius_never_beats_the_global_one(emergency_network):
    G = emergency_network
    rad = cx.radius(G)
    for u, v in G.edges:
        assert cx.local_radius(G, u, v) >= rad - 1e-9
    assert min(cx.local_radius(G, u, v) for u, v in G.edges) == pytest.approx(rad)


def test_local_center_of_an_edge(emergency_network):
    G = emergency_network
    for u, v in G.edges:
        for x in cx.local_center(G, u, v):
            a, b, lo, _ = as_interval(x)
            p = cx.Point(a, b, lo)
            assert cx.eccentricity(G, p) == pytest.approx(
                cx.local_radius(G, u, v), abs=1e-9
            )
