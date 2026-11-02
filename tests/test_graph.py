"""Construction, validation and the caching contract of ContinuousGraph."""

import networkx as nx
import pytest

import continuumx as cx
from continuumx.exceptions import (
    InvalidLengthError,
    NotConnectedError,
    UnsupportedGraphError,
)


def test_rejects_disconnected_graphs():
    g = nx.Graph([(0, 1), (2, 3)])
    with pytest.raises(NotConnectedError):
        cx.ContinuousGraph(g)


def test_rejects_empty_directed_and_multigraphs():
    with pytest.raises(UnsupportedGraphError):
        cx.ContinuousGraph(nx.Graph())
    with pytest.raises(UnsupportedGraphError):
        cx.ContinuousGraph(nx.DiGraph([(0, 1)]))
    with pytest.raises(UnsupportedGraphError):
        cx.ContinuousGraph(nx.MultiGraph([(0, 1)]))


def test_rejects_self_loops():
    g = nx.Graph([(0, 1)])
    g.add_edge(0, 0)
    with pytest.raises(UnsupportedGraphError):
        cx.ContinuousGraph(g)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_rejects_non_positive_lengths(bad):
    g = nx.Graph()
    g.add_edge(0, 1, length=bad)
    with pytest.raises(InvalidLengthError):
        cx.ContinuousGraph(g)


def test_helpful_error_when_lengths_live_in_weight():
    g = nx.Graph()
    g.add_edge(0, 1, weight=2.0)
    with pytest.raises(InvalidLengthError, match="length='weight'"):
        cx.ContinuousGraph(g)
    G = cx.ContinuousGraph(g, length="weight")
    assert G.total_length() == 2.0


def test_missing_attribute_defaults_to_unit_lengths():
    G = cx.ContinuousGraph(nx.path_graph(3))
    assert G.total_length() == 2.0
    assert G.edge_length(0, 1) == 1.0


def test_single_vertex_graph_has_trivial_parameters():
    g = nx.Graph()
    g.add_node("solo")
    G = cx.ContinuousGraph(g)
    assert cx.diameter(G) == 0.0
    assert cx.radius(G) == 0.0
    assert cx.center(G) == [cx.Point.at_vertex("solo")]
    assert cx.diametral_pair(G) == (cx.Point.at_vertex("solo"),) * 2


def test_edges_are_canonically_oriented():
    g = nx.Graph([("b", "a")])
    G = cx.ContinuousGraph(g)
    assert G.edges == [("a", "b")]
    assert G.edge_length("b", "a") == G.edge_length("a", "b")


def test_unknown_edge_raises():
    G = cx.ContinuousGraph(nx.path_graph(3))
    with pytest.raises(ValueError):
        G.edge_length(0, 2)


def test_distance_cache_is_shared_and_lazy():
    G = cx.ContinuousGraph(nx.path_graph(5))
    assert G._dist_cache == {}
    G.vertex_distance(0, 4)
    assert set(G._dist_cache) == {0}
    G.precompute()
    assert set(G._dist_cache) == set(G.nodes)


def test_as_continuous_reuses_an_existing_view():
    G = cx.ContinuousGraph(nx.path_graph(3))
    assert cx.as_continuous(G) is G

