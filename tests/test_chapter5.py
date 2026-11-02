"""The code of Chapter 5 of the master's thesis, run as written.

Chapter 5 walks through the package on the weighted emergency network and
prints the result of every call.  This module runs those calls with the same
``float`` lengths and checks the printed outputs literally, through ``repr``,
so that the chapter and the package cannot drift apart.  The mathematical
values themselves are checked exactly, with ``Fraction`` lengths, in
``test_thesis_examples.py``.
"""

import networkx as nx
import pytest

import continuumx as cx


@pytest.fixture
def emergency():
    """Section 5.2: the weighted emergency network, built as in the chapter."""
    roads = [
        ("a", "b", 1.0), ("b", "c", 3.0), ("c", "d", 1.0), ("d", "e", 1.0),
        ("e", "a", 1.0), ("a", "f", 2.0), ("c", "g", 2.0),
    ]
    g = nx.Graph()
    g.add_weighted_edges_from(roads, weight="length")
    return g, cx.ContinuousGraph(g)


# ---------------------------------------------------------------------------
# Section 5.2: building a continuous graph, points
# ---------------------------------------------------------------------------


def test_points_are_normalized():
    assert (cx.Point("b", "a", 0.25) == cx.Point("a", "b", 0.75)) is True
    assert (cx.Point("a", "b", 0.0) == cx.Point("a", "c", 0.0)) is True


def test_points_of_the_emergency_network(emergency):
    _, G = emergency
    p = G.point("b", "c", offset=1.0)
    assert repr(p) == "Point('b', 'c', 0.3333333333333333)"
    assert repr(G.offset(p)) == "1.0"

    q = G.point("d", "e", theta=0.25)
    assert repr(q) == "Point('d', 'e', 0.25)"


# ---------------------------------------------------------------------------
# Section 5.3: distances
# ---------------------------------------------------------------------------


def test_distance(emergency):
    _, G = emergency
    p = G.point("b", "c", offset=1.0)
    q = G.point("d", "e", theta=0.25)
    assert repr(cx.distance(G, p, q)) == "3.25"


# ---------------------------------------------------------------------------
# Section 5.4: eccentricity
# ---------------------------------------------------------------------------


def test_eccentricity(emergency):
    _, G = emergency
    assert repr(cx.eccentricity(G, G.point("d"))) == "4.0"
    assert repr(cx.eccentricity(G, G.point("d", "e", theta=0.5))) == "3.5"


def test_eccentricity_function(emergency):
    _, G = emergency
    f = cx.eccentricity_function(G, "d", "e")
    assert repr(f.points()) == "[(0.0, 4.0), (0.5, 3.5), (1.0, 4.0)]"
    assert repr(f(0.5)) == "3.5"


# ---------------------------------------------------------------------------
# Section 5.5: diameter
# ---------------------------------------------------------------------------


def test_diameter(emergency):
    g, G = emergency
    assert repr(cx.diameter(G)) == "7.0"
    p, q = cx.diametral_pair(G)
    assert repr((p, q)) == "(Point('f'), Point('g'))"
    assert repr(cx.distance(G, p, q)) == "7.0"
    assert repr(nx.diameter(g, weight="length")) == "7.0"


# ---------------------------------------------------------------------------
# Section 5.6: radius and center
# ---------------------------------------------------------------------------


def test_radius_and_center(emergency):
    g, G = emergency
    assert repr(cx.local_radius(G, "d", "e")) == "3.5"
    assert repr(cx.local_radius(G, "a", "f")) == "5.0"
    assert repr(cx.radius(G)) == "3.5"
    assert repr(cx.center(G)) == "[Point('d', 'e', 0.5)]"
    assert repr(nx.radius(g, weight="length")) == "4.0"
    assert repr(nx.center(g, weight="length")) == "['d', 'e']"


def test_center_of_a_cycle_is_four_segments():
    """Remark 5.1: on the continuous cycle C_4 with unit edges, ``cx.center``
    returns four segments, each covering an entire edge."""
    center = cx.center(nx.cycle_graph(4))
    assert len(center) == 4
    assert all(isinstance(x, cx.Segment) for x in center)
    assert all((x.theta_min, x.theta_max) == (0.0, 1.0) for x in center)
    assert len({frozenset((x.u, x.v)) for x in center}) == 4
