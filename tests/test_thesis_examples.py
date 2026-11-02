"""The worked examples of Chapter 4 of the master's thesis, computed with the
library.

Every number below is quoted from Chapter 4 and checked against
:mod:`continuumx`, with ``Fraction`` lengths and zero tolerance.  Each test
names the example or result it checks by its label in the LaTeX source of the
master's thesis.

The discrete values quoted for comparison (eccentricities, radius, ...) are
computed with NetworkX on the same weighted graph.
"""

import random
from fractions import Fraction as F
from itertools import combinations

import networkx as nx
import pytest

import continuumx as cx

# ---------------------------------------------------------------------------
# The graphs of the master's thesis
# ---------------------------------------------------------------------------


def make(edges):
    g = nx.Graph()
    for u, v, ell in edges:
        g.add_edge(u, v, length=F(ell))
    return cx.ContinuousGraph(g, tol=0)


def disc(G, v):
    """Discrete eccentricity of ``v`` in the weighted graph underlying ``G``."""
    return nx.eccentricity(G.graph, v, weight="length")


@pytest.fixture
def running():
    """The graph of ``fig:continuous-construction`` (Sections 4.1-4.2)."""
    return make([("a", "b", 3), ("b", "c", 2), ("a", "c", 4), ("c", "d", 2)])


@pytest.fixture
def logistics():
    """The weighted logistics network (``ex:logistics-weighted``)."""
    return make(
        [("a", "b", 10), ("b", "c", 5), ("a", "c", 20), ("a", "d", 5), ("d", "e", 15)]
    )


@pytest.fixture
def emergency():
    """The weighted emergency network (``ex:emergency``)."""
    return make(
        [
            ("a", "b", 1),
            ("b", "c", 3),
            ("c", "d", 1),
            ("d", "e", 1),
            ("e", "a", 1),
            ("a", "f", 2),
            ("c", "g", 2),
        ]
    )


@pytest.fixture
def tree():
    """The complete binary tree of depth 3 (``example_eccentricity``)."""
    t = nx.relabel_nodes(
        nx.balanced_tree(2, 3), {i: "abcdefghijklmno"[i] for i in range(15)}
    )
    nx.set_edge_attributes(t, F(1), "length")
    return cx.ContinuousGraph(t, tol=0)


def cycle(n):
    g = nx.cycle_graph(n)
    nx.set_edge_attributes(g, F(1), "length")
    return cx.ContinuousGraph(g, tol=0)


@pytest.fixture
def k4():
    return make([(i, j, 1) for i, j in combinations("abcd", 2)])


# ---------------------------------------------------------------------------
# Section 4.2: paths and distances
# ---------------------------------------------------------------------------


class TestDistances:
    def test_continuous_distance(self, running):
        """ex:continuous-distance: d(p,q) = 5."""
        p = running.point("a", "b", offset=1)
        q = running.point("c", "d", offset=1)
        assert cx.distance(running, p, q) == 5

    def test_computing_a_continuous_distance(self, running):
        """ex:distance-formulas: theta = 1/3, mu = 1/2."""
        p = running.point("a", "b", offset=1)
        q = running.point("c", "d", offset=1)
        assert p.theta == F(1, 3) and q.theta == F(1, 2)
        assert cx.distance_to_vertex(running, p, "d") == 6
        assert cx.distance_to_vertex(running, p, "c") == 4
        assert cx.distance(running, p, q) == 5

    def test_distance_is_symmetric_here(self, running):
        p = running.point("a", "b", offset=1)
        q = running.point("c", "d", offset=1)
        assert cx.distance(running, q, p) == cx.distance(running, p, q)

    def test_distance_within_an_edge(self, running):
        """ex:same-edge: x, y in ac with l(a,x) = 1, l(a,y) = 7/2."""
        x = running.point("a", "c", offset=1)
        y = running.point("a", "c", offset=F(7, 2))
        assert (x.theta, y.theta) == (F(1, 4), F(7, 8))
        assert cx.distance(running, x, y) == F(5, 2)

    def test_same_edge_leaving_the_edge_is_shorter(self):
        """The remark after ex:same-edge: if l(ac) exceeds l(ab) + l(bc) = 5,
        the route through b beats the edge itself."""
        G = make([("a", "b", 3), ("b", "c", 2), ("a", "c", 8)])
        x = G.point("a", "c", offset=F(1, 2))
        y = G.point("a", "c", offset=F(15, 2))
        # direct: 7; leaving through a and re-entering through c: 1/2 + 5 + 1/2.
        assert cx.distance(G, x, y) == 6
        assert cx.distance(G, x, y) < G.edge_length("a", "c") - 1


# ---------------------------------------------------------------------------
# Section 4.3.1: eccentricities
# ---------------------------------------------------------------------------


class TestEccentricities:
    def test_tree_vertices_keep_their_discrete_eccentricity(self, tree):
        """ex:tree-ecc-cont: leaves are vertices, so ecc_T = ecc_G."""
        for v in tree.nodes:
            assert cx.eccentricity(tree, tree.point(v)) == disc(tree, v)

    @pytest.mark.parametrize("t", [F(0), F(1, 4), F(1, 2), F(3, 4), F(1)])
    def test_tree_interior_point(self, tree, t):
        """ex:tree-ecc-cont: x on dh at distance t from the leaf h has
        ecc(x) = 6 - t."""
        x = tree.point("d", "h", offset=1 - t)  # offset is measured from d
        assert cx.eccentricity(tree, x) == 6 - t

    def test_farthest_leaves_from_d(self, tree):
        """ex:tree-ecc-cont: the leaves l, m, n, o outside dh are at distance 5."""
        d = tree.point("d")
        assert {v: cx.distance(tree, d, tree.point(v)) for v in "lmno"} == {
            v: 5 for v in "lmno"
        }

    def test_cycles(self):
        """ex:cycle-ecc-cont: ecc(v1) = P/2, against 2 and 3 in the discrete case."""
        c5, c6 = cycle(5), cycle(6)
        assert cx.eccentricity(c5, c5.point(0)) == F(5, 2)
        assert cx.eccentricity(c6, c6.point(0)) == 3
        assert disc(c5, 0) == 2 and disc(c6, 0) == 3
        # every point has the same eccentricity
        for G, n in ((c5, 5), (c6, 6)):
            for t in (F(0), F(1, 3), F(1, 2), F(5, 7)):
                assert cx.eccentricity(G, G.point(0, 1, theta=t)) == F(n, 2)

    def test_logistics_discrete_eccentricities(self, logistics):
        """ex:logistics-ecc-cont: the quoted discrete eccentricities."""
        assert {v: disc(logistics, v) for v in "abcde"} == {
            "a": 20,
            "b": 30,
            "c": 35,
            "d": 20,
            "e": 35,
        }

    def test_logistics_edge_is_slack(self, logistics):
        """ex:logistics-ecc-cont: l(ac) = 20 exceeds d(a,c) = 15."""
        assert logistics.edge_length("a", "c") == 20
        assert logistics.vertex_distance("a", "c") == 15

    @pytest.mark.parametrize("theta", [F(0), F(1, 4), F(1, 2), F(7, 8), F(1)])
    def test_logistics_distance_to_e_along_ac(self, logistics, theta):
        """ex:logistics-ecc-cont: d(x,e) = min{20 theta + 20, 55 - 20 theta}."""
        x = logistics.point("a", "c", theta=theta)
        want = min(20 * theta + 20, 55 - 20 * theta)
        assert cx.distance(logistics, x, logistics.point("e")) == want

    def test_logistics_peak_point(self, logistics):
        """ex:logistics-ecc-cont: x* at theta = 7/8, with ecc(x*) = 37.5."""
        xs = logistics.point("a", "c", theta=F(7, 8))
        dist = {v: cx.distance(logistics, xs, logistics.point(v)) for v in "abcde"}
        assert dist == {
            "a": F(35, 2),
            "b": F(15, 2),
            "c": F(5, 2),
            "d": F(45, 2),
            "e": F(75, 2),
        }
        assert cx.eccentricity(logistics, xs) == F(75, 2)

    @pytest.mark.parametrize("theta", [F(0), F(1, 4), F(7, 8), F(1)])
    def test_logistics_distance_from_d_along_ac(self, logistics, theta):
        """ex:logistics-ecc-cont: d(d,x) = min{5 + 20 theta, 40 - 20 theta}."""
        x = logistics.point("a", "c", theta=theta)
        want = min(5 + 20 * theta, 40 - 20 * theta)
        assert cx.distance(logistics, logistics.point("d"), x) == want

    def test_logistics_vertex_eccentricities(self, logistics):
        """ex:logistics-ecc-cont: a, b, c agree with the discrete ones; d and e
        do not (22.5 and 37.5)."""
        got = {v: cx.eccentricity(logistics, logistics.point(v)) for v in "abcde"}
        assert got == {"a": 20, "b": 30, "c": 35, "d": F(45, 2), "e": F(75, 2)}

    def test_emergency_vertices_keep_their_discrete_eccentricity(self, emergency):
        """ex:emergency-ecc-cont and ex:emergency-ecc."""
        got = {v: cx.eccentricity(emergency, emergency.point(v)) for v in "abcdefg"}
        assert got == {"a": 5, "b": 5, "c": 5, "d": 4, "e": 4, "f": 7, "g": 7}
        assert got == {v: disc(emergency, v) for v in "abcdefg"}

    def test_emergency_unweighted_eccentricities(self):
        """ex:emergency-ecc, unweighted case (a discrete example)."""
        g = nx.Graph(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
                ("d", "e"),
                ("e", "a"),
                ("a", "f"),
                ("c", "g"),
            ]
        )
        assert nx.eccentricity(g) == {
            "a": 3,
            "b": 2,
            "c": 3,
            "d": 3,
            "e": 3,
            "f": 4,
            "g": 4,
        }

    @pytest.mark.parametrize(
        "theta", [F(0), F(1, 8), F(1, 4), F(1, 2), F(3, 4), F(7, 8), F(1)]
    )
    def test_emergency_interior_points(self, emergency, theta):
        """ex:emergency-ecc-cont: ecc(x) = max{4 - theta, 3 + theta, 7/2}."""
        x = emergency.point("d", "e", theta=theta)
        want = max(4 - theta, 3 + theta, F(7, 2))
        assert cx.eccentricity(emergency, x) == want

    def test_emergency_midpoint_beats_every_junction(self, emergency):
        """ex:emergency-ecc-cont: 3.5 at the midpoint of de, against 4."""
        mid = emergency.point("d", "e", theta=F(1, 2))
        assert cx.eccentricity(emergency, mid) == F(7, 2)
        assert (
            min(cx.eccentricity(emergency, emergency.point(v)) for v in "abcdefg") == 4
        )

    def test_emergency_cycle_is_within_half_the_perimeter(self, emergency):
        """ex:emergency-ecc-cont: P = 7, so every cycle point is within 7/2."""
        x = emergency.point("d", "e", theta=F(1, 3))
        for v in "abcde":
            assert cx.distance(emergency, x, emergency.point(v)) <= F(7, 2)
        assert cx.distance(emergency, emergency.point("f"), emergency.point("g")) == 7


# ---------------------------------------------------------------------------
# Section 4.3.2: diameter
# ---------------------------------------------------------------------------


class TestDiameter:
    def test_tree(self, tree):
        """ex:tree-diam-cont: diam = 6, with the same 16 pairs of leaves."""
        assert cx.diameter(tree) == 6
        leaves = "hijklmno"
        pairs = [
            (u, v)
            for u, v in combinations(leaves, 2)
            if cx.distance(tree, tree.point(u), tree.point(v)) == 6
        ]
        assert len(pairs) == 16
        assert nx.diameter(tree.graph) == 6

    def test_cycles(self):
        """ex:cycle-diam-cont: diam(C6) = 3, diam(C5) = 5/2 > diam(C_5) = 2."""
        c5, c6 = cycle(5), cycle(6)
        assert cx.diameter(c6) == 3 and cx.diameter(c5) == F(5, 2)
        assert nx.diameter(c5.graph) == 2 and nx.diameter(c6.graph) == 3

    def test_cycle_antipodal_pairs(self):
        """ex:cycle-diam-cont: every point forms a diametral pair with its
        antipodal point.  Here p lies on the edge 01 of C5 and its antipode on
        the edge 23, at the same relative position."""
        c5 = cycle(5)
        for t in (F(0), F(1, 5), F(1, 2), F(4, 5)):
            p = c5.point(0, 1, theta=t)
            q = (
                c5.point(2, 3, theta=t + F(1, 2))
                if t <= F(1, 2)
                else c5.point(3, 4, theta=t - F(1, 2))
            )
            assert cx.distance(c5, p, q) == F(5, 2)

    def test_logistics(self, logistics):
        """ex:logistics-diam-cont: diam = 37.5 > 35, pair (x*, e)."""
        assert cx.diameter(logistics) == F(75, 2)
        assert nx.diameter(logistics.graph, weight="length") == 35
        p, q = cx.diametral_pair(logistics)
        xs = logistics.point("a", "c", theta=F(7, 8))
        e = logistics.point("e")
        assert (logistics.same_point(p, xs) and logistics.same_point(q, e)) or (
            logistics.same_point(q, xs) and logistics.same_point(p, e)
        )
        assert cx.distance(logistics, xs, e) == F(75, 2)

    def test_logistics_diametral_pair_is_unique(self, logistics):
        """ex:logistics-diam-cont: only x* and e have eccentricity 37.5."""
        diam = cx.diameter(logistics)
        extreme = {}
        for u, v in logistics.edges:
            f = cx.eccentricity_function(logistics, u, v)
            at_max = [x for x, y in f.points() if y == diam]
            if at_max:
                extreme[(u, v)] = at_max
        assert extreme == {("a", "c"): [F(7, 8)], ("d", "e"): [F(1)]}

    def test_logistics_diametral_paths_tie(self, logistics):
        """ex:logistics-diam-cont: (x*,a,d,e) and (x*,c,b,a,d,e) both have
        length 37.5."""
        G = logistics
        via_a = F(35, 2) + G.vertex_distance("a", "e")
        via_c = F(5, 2) + G.vertex_distance("c", "e")
        assert via_a == via_c == F(75, 2)

    def test_emergency(self, emergency):
        """ex:emergency-diam-cont: diam = 7 = discrete, pair (f, g)."""
        assert cx.diameter(emergency) == 7
        assert nx.diameter(emergency.graph, weight="length") == 7
        p, q = cx.diametral_pair(emergency)
        assert {p, q} == {emergency.point("f"), emergency.point("g")}


# ---------------------------------------------------------------------------
# Section 4.3.3: radius and center
# ---------------------------------------------------------------------------


def central_points(G):
    """The center as a list of points (every example below has finitely many)."""
    pts = cx.center(G)
    for x in pts:
        assert isinstance(x, cx.Point), f"expected an isolated point, got {x}"
    return pts


class TestRadiusAndCenter:
    def test_tree(self, tree):
        """ex:tree-radius-cont: rad = 3 and the center is {a}."""
        assert cx.radius(tree) == 3 == nx.radius(tree.graph)
        (c,) = central_points(tree)
        assert tree.same_point(c, tree.point("a"))

    def test_cycles(self):
        """ex:cycle-radius-cont: rad(C6) = 3, rad(C5) = 5/2, center = the
        whole cycle; rad(C_5) = 2."""
        for n, rad in ((5, F(5, 2)), (6, 3)):
            G = cycle(n)
            assert cx.radius(G) == rad
            segs = cx.center(G)
            assert len(segs) == n
            assert all(s.theta_min == 0 and s.theta_max == 1 for s in segs)
        assert nx.radius(cycle(5).graph) == 2

    def test_logistics_lower_bound_and_radius(self, logistics):
        """ex:logistics-radius-cont: rad(G) = 20 (centers a, d), rad(calG) = 18.75."""
        assert nx.radius(logistics.graph, weight="length") == 20
        assert set(nx.center(logistics.graph, weight="length")) == {"a", "d"}
        assert cx.diameter(logistics) / 2 == F(75, 4)
        assert cx.radius(logistics) == F(75, 4)
        assert cx.diameter(logistics) == 2 * cx.radius(logistics)

    def test_logistics_center_is_y(self, logistics):
        """ex:logistics-radius-cont: y in ad at distance 1.25 from a is the
        only central point."""
        (c,) = central_points(logistics)
        y = logistics.point("a", "d", offset=F(5, 4))
        assert y.theta == F(1, 4)
        assert logistics.same_point(c, y)
        xs = logistics.point("a", "c", theta=F(7, 8))
        e = logistics.point("e")
        assert cx.distance(logistics, y, xs) == cx.distance(logistics, y, e) == F(75, 4)
        assert cx.eccentricity(logistics, y) == F(75, 4)

    @pytest.mark.parametrize("theta", [F(0), F(1, 8), F(1, 4), F(1, 2), F(1)])
    def test_logistics_envelope_on_ad(self, logistics, theta):
        """ex:logistics-radius-cont: ecc(x) = max{17.5 + 5 theta, 20 - 5 theta}
        on ad."""
        x = logistics.point("a", "d", theta=theta)
        want = max(F(35, 2) + 5 * theta, 20 - 5 * theta)
        assert cx.eccentricity(logistics, x) == want

    def test_logistics_relative_eccentricities_on_ad(self, logistics):
        """ex:logistics-radius-cont: ecc(x,ac) and ecc(x,de) along ad."""
        for theta in (F(0), F(1, 4), F(1, 2), F(1)):
            x = logistics.point("a", "d", theta=theta)
            assert (
                cx.eccentricity_to_edge(logistics, x, "a", "c") == F(35, 2) + 5 * theta
            )
            assert cx.eccentricity_to_edge(logistics, x, "d", "e") == 20 - 5 * theta

    def test_emergency(self, emergency):
        """ex:emergency-radius-cont: rad = 7/2 < 4 = rad(G), center = the
        midpoint of de, against {d, e} in the discrete case."""
        assert cx.radius(emergency) == F(7, 2)
        assert nx.radius(emergency.graph, weight="length") == 4
        assert set(nx.center(emergency.graph, weight="length")) == {"d", "e"}
        assert cx.center(emergency) == [cx.Point("d", "e", F(1, 2))]  # as in Ch. 5
        (c,) = central_points(emergency)
        assert emergency.same_point(c, emergency.point("d", "e", theta=F(1, 2)))
        assert cx.diameter(emergency) == 2 * cx.radius(emergency) == 7

    def test_emergency_center_unique_by_the_proof(self, emergency):
        """ex:emergency-radius-cont: a central point is at distance 3/2 from
        a and from c; the candidates on the other roads fail."""
        G = emergency
        x = G.point("d", "e", theta=F(1, 2))
        assert (
            cx.distance(G, x, G.point("a"))
            == cx.distance(G, x, G.point("c"))
            == F(3, 2)
        )
        on_cb = G.point("b", "c", offset=F(1, 2))  # 1/2 from b
        on_af = G.point("a", "f", offset=F(3, 2))
        assert cx.distance(G, on_cb, G.point("a")) == F(3, 2)
        assert cx.distance(G, on_af, G.point("a")) == F(3, 2)
        assert cx.distance(G, on_cb, G.point("c")) != F(3, 2)
        assert cx.distance(G, on_af, G.point("c")) != F(3, 2)


# ---------------------------------------------------------------------------
# Section 4.3.4: basic properties
# ---------------------------------------------------------------------------


class TestBasicProperties:
    def test_complete_graph(self, k4):
        """ex:complete-cont: rad = 3/2, diam = 2, center = V, not self-centered."""
        assert nx.radius(k4.graph) == nx.diameter(k4.graph) == 1
        for v in "abcd":
            assert cx.eccentricity(k4, k4.point(v)) == F(3, 2)
        assert cx.radius(k4) == F(3, 2) and cx.diameter(k4) == 2
        center = central_points(k4)
        assert len(center) == 4
        for v in "abcd":
            assert any(k4.same_point(c, k4.point(v)) for c in center)

    @pytest.mark.parametrize("theta", [F(0), F(1, 8), F(1, 4), F(1, 2)])
    def test_complete_graph_interior_point(self, k4, theta):
        """ex:complete-cont: ecc(p) = theta + 3/2 for theta <= 1/2."""
        p = k4.point("a", "b", theta=theta)
        assert cx.eccentricity(k4, p) == theta + F(3, 2)

    def test_complete_graph_farthest_point_is_a_midpoint(self, k4):
        """ex:complete-cont: from a vertex, the farthest point of an edge not
        containing it is the midpoint, at distance 3/2."""
        far = cx.farthest_point_on_edge(k4, k4.point("a"), "c", "d")
        assert far.theta == F(1, 2)
        assert cx.distance(k4, k4.point("a"), far) == F(3, 2)

    def test_radius_is_not_comparable_with_the_discrete_one(self, tree, emergency):
        """rem:radius-incomparable."""
        assert cx.radius(tree) == nx.radius(tree.graph) == 3
        assert cx.radius(emergency) < nx.radius(emergency.graph, weight="length")
        c3 = cycle(3)
        assert nx.radius(c3.graph) == 1 and cx.radius(c3) == F(3, 2)

    def test_not_geometric_example_k4_is_unit_lengths(self, k4):
        """ex:not-geometric: K4 with all lengths 1 (six edges, total length 6)."""
        assert k4.number_of_edges() == 6 and k4.total_length() == 6

    def test_inequalities_hold_on_every_example(
        self, logistics, emergency, tree, k4, running
    ):
        """prop:rad_diam_continuous and prop:discrete-vs-continuous."""
        for G in (logistics, emergency, tree, k4, running, cycle(5), cycle(6)):
            rad, diam = cx.radius(G), cx.diameter(G)
            assert rad <= diam <= 2 * rad
            drad = nx.radius(G.graph, weight="length")
            ddiam = nx.diameter(G.graph, weight="length")
            assert ddiam <= diam
            assert ddiam <= 2 * rad
            assert drad <= 2 * rad


# ---------------------------------------------------------------------------
# Propositions, on random graphs
# ---------------------------------------------------------------------------


def random_lengths(rng, g):
    for u, v in g.edges:
        g[u][v]["length"] = F(rng.randint(1, 12), rng.choice([1, 2, 3]))
    return cx.ContinuousGraph(g, tol=0)


class TestPropositions:
    def test_trees(self):
        """prop:tree-diam-cont and prop:tree-radius-cont: diam is the discrete
        diameter, rad = diam / 2 and the center is the midpoint of a diametral
        path."""
        rng = random.Random(11)
        for _ in range(25):
            n = rng.randint(3, 10)
            g = (
                nx.random_labeled_tree(n, seed=rng.randint(0, 10**6))
                if hasattr(nx, "random_labeled_tree")
                else nx.random_tree(n, seed=rng.randint(0, 10**6))
            )
            G = random_lengths(rng, g)
            diam = cx.diameter(G)
            assert diam == nx.diameter(G.graph, weight="length")
            assert cx.radius(G) == diam / 2
            (c,) = central_points(G)
            assert cx.eccentricity(G, c) == diam / 2
            p, q = cx.diametral_pair(G)
            assert G.vertex_distance(p.vertex, q.vertex) == diam  # both are leaves

    def test_cycles(self):
        """prop:continuous_cycle_diameter: diam = rad = P/2, center = G."""
        rng = random.Random(5)
        for _ in range(25):
            n = rng.randint(3, 9)
            G = random_lengths(rng, nx.cycle_graph(n))
            half = G.total_length() / 2
            assert cx.diameter(G) == cx.radius(G) == half
            assert len(cx.center(G)) == n
            for t in (F(0), F(1, 3), F(2, 3)):
                assert cx.eccentricity(G, G.point(0, 1, theta=t)) == half

    def test_general_graphs(self):
        """prop:rad_diam_continuous and prop:discrete-vs-continuous."""
        rng = random.Random(3)
        for _ in range(40):
            n = rng.randint(3, 8)
            while True:
                g = nx.gnp_random_graph(
                    n, rng.uniform(0.3, 0.9), seed=rng.randint(0, 10**6)
                )
                if nx.is_connected(g):
                    break
            G = random_lengths(rng, g)
            rad, diam = cx.radius(G), cx.diameter(G)
            ddiam = nx.diameter(G.graph, weight="length")
            drad = nx.radius(G.graph, weight="length")
            assert rad <= diam <= 2 * rad
            assert ddiam <= diam and ddiam <= 2 * rad and drad <= 2 * rad
            # the bounds with l_max (older form of prop:discrete-vs-continuous)
            lmax = max(d["length"] for *_, d in G.graph.edges(data=True))
            assert diam <= ddiam + lmax
            assert rad <= drad + lmax / 2


# ---------------------------------------------------------------------------
# Section 4.4: algorithmic computation
# ---------------------------------------------------------------------------


def hat_of(G, edge, witness):
    """The hat of ``edge`` associated with the edge ``witness``, as (b, s, h)."""
    out = [
        (h.b, h.s, h.h) for h, w in cx.hat_functions(G, *edge) if set(w) == set(witness)
    ]
    return out


class TestAlgorithmicComputation:
    def test_farthest_point_of_an_edge(self, logistics):
        """ex:farthest-point: from b, the farthest point of ac is at
        distance 17.5, at theta = 3/8 (7.5 from a, 12.5 from c)."""
        b = logistics.point("b")
        assert cx.eccentricity_to_edge(logistics, b, "a", "c") == F(35, 2)
        x = cx.farthest_point_on_edge(logistics, b, "a", "c")
        assert x.theta == F(3, 8)
        assert logistics.edge_length("a", "c") * x.theta == F(15, 2)
        assert cx.distance(logistics, b, x) == F(35, 2)

    def test_hat_functions_on_ad(self, logistics):
        """ex:logistics-hats: the table of the six hats of the edge ad."""
        table = {
            ("a", "c"): [(1, 1, F(45, 2))],
            ("d", "e"): [(0, 0, 20)],
            ("b", "c"): [(1, 1, 20)],
            ("a", "b"): [(1, 1, 15)],
            ("a", "d"): [(1, 1, 5), (0, 0, 5)],
        }
        for witness, want in table.items():
            got = sorted(hat_of(logistics, ("a", "d"), witness))
            assert got == sorted((F(b), F(s), F(h)) for b, s, h in want), witness
        # all of them have slope l(ad) = 5
        assert {h.slope for h, _ in cx.hat_functions(logistics, "a", "d")} == {5}

    def test_upper_envelope_on_ad(self, logistics):
        """ex:logistics-hats: the envelope is max{17.5 + 5 theta, 20 - 5 theta}."""
        f = cx.eccentricity_function(logistics, "a", "d")
        for theta in (F(0), F(1, 8), F(1, 4), F(1, 2), F(1)):
            assert f(theta) == max(F(35, 2) + 5 * theta, 20 - 5 * theta)
        value, where = f.minimum()
        assert value == F(75, 4) and where == [(F(1, 4), F(1, 4))]

    def test_maximum_distance_between_two_edges(self, emergency):
        """ex:max-edge-edge: dist_max(bc, de) = 7/2, attained on bc exactly on
        the interval [b_wz, s_wz] = [1/6, 1/2]."""
        ((b, s, h),) = hat_of(emergency, ("b", "c"), ("d", "e"))
        assert (b, s, h) == (F(1, 6), F(1, 2), F(7, 2))
        assert emergency.edge_length("b", "c") * b == F(1, 2)
        assert emergency.edge_length("b", "c") * s == F(3, 2)

    def test_the_farthest_point_moves_along_de(self, emergency):
        """ex:max-edge-edge: the farthest point of de moves from d to e."""
        G = emergency
        ends = [
            cx.farthest_point_on_edge(G, G.point("b", "c", offset=o), "d", "e")
            for o in (F(1, 2), F(1), F(3, 2))
        ]
        assert G.same_point(ends[0], G.point("d"))
        assert G.same_point(ends[1], G.point("d", "e", theta=F(1, 2)))
        assert G.same_point(ends[2], G.point("e"))
        for o in (F(1, 2), F(1), F(3, 2)):
            x = G.point("b", "c", offset=o)
            assert cx.eccentricity_to_edge(G, x, "d", "e") == F(7, 2)

    def test_diameter_algorithm_on_the_emergency_network(self, emergency):
        """ex:emergency-diam-alg: the values of dist_max over pairs of edges."""
        G = emergency
        dmax = {}
        for u, v in G.edges:
            for h, w in cx.hat_functions(G, u, v):
                dmax[frozenset([frozenset((u, v)), frozenset(w)])] = h.h

        def pair(e1, e2):
            return dmax[frozenset([frozenset(e1), frozenset(e2)])]

        assert G.number_of_edges() == 7
        assert pair("af", "cg") == 7
        assert pair("af", "bc") == pair("ab", "cg") == F(11, 2)
        # no other pair of distinct edges reaches 11/2 ...
        others = [
            d
            for k, d in dmax.items()
            if len(k) == 2
            and k
            not in (
                frozenset([frozenset("af"), frozenset("cg")]),
                frozenset([frozenset("af"), frozenset("bc")]),
                frozenset([frozenset("ab"), frozenset("cg")]),
            )
        ]
        assert max(others) < F(11, 2)
        # ... and two edges of the cycle abcde are within half its perimeter.
        cyc = ["ab", "bc", "cd", "de", "ea"]
        assert max(pair(e1, e2) for e1, e2 in combinations(cyc, 2)) == F(7, 2)

    def test_pendant_interval_collapses(self, emergency):
        """ex:emergency-diam-alg: on af, the interval [b_wz, s_wz] of cg reduces
        to theta = 1 (the vertex f), and the farthest point of cg is then g."""
        ((b, s, h),) = hat_of(emergency, ("a", "f"), ("c", "g"))
        assert (b, s) == (1, 1) and h == 7
        far = cx.farthest_point_on_edge(emergency, emergency.point("f"), "c", "g")
        assert emergency.same_point(far, emergency.point("g"))

    def test_hats_of_the_edge_de(self, emergency):
        """ex:emergency-radius-alg: the table of hats (b, s, h) of the edge de."""
        want = {
            ("d", "e"): sorted([(F(1), F(1), F(1)), (F(0), F(0), F(1))]),
            ("c", "d"): [(F(1), F(1), F(2))],
            ("a", "e"): [(F(0), F(0), F(2))],
            ("a", "b"): [(F(0), F(0), F(3))],
            ("b", "c"): [(F(0), F(1), F(7, 2))],
            ("a", "f"): [(F(0), F(0), F(4))],
            ("c", "g"): [(F(1), F(1), F(4))],
        }
        for witness, hats in want.items():
            assert sorted(hat_of(emergency, ("d", "e"), witness)) == hats, witness

    def test_envelope_of_the_edge_de(self, emergency):
        """ex:emergency-radius-alg: the final envelope is max{4 - theta, 3 + theta},
        with minimum 7/2 at the single breakpoint theta = 1/2."""
        f = cx.eccentricity_function(emergency, "d", "e")
        for theta in (F(0), F(1, 4), F(1, 2), F(3, 4), F(1)):
            assert f(theta) == max(4 - theta, 3 + theta)
        assert f.minimum() == (F(7, 2), [(F(1, 2), F(1, 2))])
        assert cx.local_radius(emergency, "d", "e") == F(7, 2)
        assert cx.local_center(emergency, "d", "e") == [cx.Point("d", "e", F(1, 2))]
