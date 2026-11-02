"""Canonical form, equality and hashing of points and segments."""

import networkx as nx
import pytest

import continuumx as cx
from continuumx.exceptions import InvalidPointError


def test_reversed_triples_are_the_same_point():
    assert cx.Point("b", "a", 0.25) == cx.Point("a", "b", 0.75)
    assert hash(cx.Point("b", "a", 0.25)) == hash(cx.Point("a", "b", 0.75))


def test_endpoints_normalize_to_vertices():
    assert cx.Point("a", "b", 0.0) == cx.Point.at_vertex("a")
    assert cx.Point("a", "b", 1.0) == cx.Point.at_vertex("b")
    # A vertex reached through two different incident edges is one point.
    assert cx.Point("a", "b", 0.0) == cx.Point("a", "c", 0.0)


def test_vertex_flags():
    p = cx.Point.at_vertex("a")
    assert p.is_vertex and p.vertex == "a" and p.edge is None
    q = cx.Point("a", "b", 0.5)
    assert not q.is_vertex and q.vertex is None and q.edge == ("a", "b")


def test_points_are_usable_in_sets():
    assert len({cx.Point("a", "b", 0.5), cx.Point("b", "a", 0.5)}) == 1


def test_theta_out_of_range():
    with pytest.raises(InvalidPointError):
        cx.Point("a", "b", 1.5)
    with pytest.raises(InvalidPointError):
        cx.Point("a", "b", -0.1)


def test_mixed_node_types_are_orientable():
    # node_key gives a total order even when the nodes are not comparable.
    assert cx.Point(0, "a", 0.25) == cx.Point("a", 0, 0.75)


def test_segment_normalization():
    s = cx.Segment("b", "a", 0.75, 0.25)
    assert (s.u, s.v) == ("a", "b")
    assert (s.theta_min, s.theta_max) == (0.25, 0.75)
    assert s.length_fraction == 0.5


def test_degenerate_segment_is_a_point():
    assert cx.Segment("a", "b", 0.5, 0.5).is_point
    assert cx.Segment("a", "b", 0.0, 0.0) == cx.Segment("a", "a", 0.0, 0.0)


def test_segment_endpoints_round_trip():
    s = cx.Segment("a", "b", 0.2, 0.8)
    p, q = s.endpoints()
    assert p == cx.Point("a", "b", 0.2)
    assert q == cx.Point("a", "b", 0.8)


def test_graph_point_builder_accepts_offsets():
    g = nx.Graph()
    g.add_edge("a", "b", length=4.0)
    g.add_edge("b", "c", length=1.0)
    G = cx.ContinuousGraph(g)
    assert G.point("a", "b", offset=1.0) == cx.Point("a", "b", 0.25)
    assert G.offset(cx.Point("a", "b", 0.25)) == 1.0
    assert G.point("c").is_vertex
    with pytest.raises(InvalidPointError):
        G.point("a", "b", offset=9.0)
