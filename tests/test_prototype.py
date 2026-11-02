"""End-to-end prototype: every step of a typical session, on one graph.

The network is a cycle ``b c d e b`` of perimeter 9 with a pendant edge ``ab``
of length 5.  Every expected value below is derived by hand in the comments,
so the test also documents *why* each number is what it is.

Lengths are :class:`fractions.Fraction` and the graph is built with ``tol=0``,
so every comparison is an exact equality.
"""

from fractions import Fraction as F

import networkx as nx

import continuumx as cx

EDGES = [
    ("a", "b", 5),
    ("b", "c", 1),
    ("b", "e", 1),
    ("c", "d", 5),
    ("d", "e", 2),
]


def test_full_session():
    # ------------------------------------------------------------------
    # 1. Build the continuous graph.
    # ------------------------------------------------------------------
    g = nx.Graph() # start with an empty graph.
    for u, v, ell in EDGES: # add edges to the graph and their lengths
        g.add_edge(u, v, length=F(ell))
    G = cx.ContinuousGraph(g, tol=0) # create a continuous graph with the given graph and tolerance

    assert G.number_of_nodes() == 5     # a,b,c,d,e
    assert G.number_of_edges() == 5     # ab, bc, be, cd, de
    assert G.total_length() == 14       # 5 + 1 + 1 + 5 + 2 = 14
    assert G.edge_length("d", "c") == 5 # edges are undirected (d,c) == (c,d)

    # ------------------------------------------------------------------
    # 2. Distances between vertices agree with networkx.
    # ------------------------------------------------------------------
    nx_dist = dict(nx.all_pairs_dijkstra_path_length(g, weight="length"))
    for u in g:
        for v in g:
            p, q = G.point(u), G.point(v)
            assert cx.distance(G, p, q) == nx_dist[u][v]
            assert G.vertex_distance(u, v) == nx_dist[u][v]

    # ------------------------------------------------------------------
    # 3. Distances between interior points.
    # ------------------------------------------------------------------
    # Same edge, where the path must leave the edge: on cd (length 5), the
    # points at 1/5 from each end are 23/5 apart along the edge, but only
    # 1/5 + d(c, d) + 1/5 = 1/5 + 4 + 1/5 = 22/5 apart through c b e d.
    p = G.point("c", "d", offset=F(1, 5))
    q = G.point("c", "d", offset=F(24, 5))
    assert cx.distance(G, p, q) == F(22, 5)
    assert cx.distance(G, p, q) < abs(G.offset(p) - G.offset(q))

    # Different edges: 1 from b on ab, 1 from d on cd.  Through c the path
    # is 1 + 1 + 4 = 6; through e it is 1 + 1 + 2 + 1 = 5.
    p = G.point("a", "b", offset=F(4))
    q = G.point("c", "d", offset=F(4))
    assert cx.distance(G, p, q) == 5

    # The two parametrizations name the same point, in either orientation.
    assert G.point("a", "b", theta=F(4, 5)) == p
    assert G.point("b", "a", offset=F(1)) == p

    # Metric axioms on a handful of points.
    sample = [G.point("a"), p, q, G.point("d", "e", theta=F(1, 2)), G.point("e")]
    for x in sample:
        assert cx.distance(G, x, x) == 0
        for y in sample:
            assert cx.distance(G, x, y) == cx.distance(G, y, x)
            for z in sample:
                assert cx.distance(G, x, z) <= cx.distance(G, x, y) + cx.distance(G, y, z)

    # ------------------------------------------------------------------
    # 4. Diameter.
    # ------------------------------------------------------------------
    # The farthest point from a is the antipode of b on the cycle, 9/2 away
    # from b: d(a, ·) = 5 + 9/2 = 19/2.  The discrete diameter is only
    # d(a, d) = 5 + 3 = 8, because the antipode is not a vertex.
    diam = cx.diameter(G)
    assert diam == F(19, 2)
    assert nx.diameter(g, weight="length") == 8
    assert diam >= nx.diameter(g, weight="length")

    # ------------------------------------------------------------------
    # 5. Diametral pair.
    # ------------------------------------------------------------------
    # The antipode of b lies on cd, 7/2 from c (b c is 1, then 7/2 more),
    # that is, at theta = 7/10.
    p, q = cx.diametral_pair(G)
    assert cx.distance(G, p, q) == diam
    assert {p, q} == {G.point("a"), G.point("c", "d", theta=F(7, 10))}

    # ------------------------------------------------------------------
    # 6. Radius.
    # ------------------------------------------------------------------
    # A point at distance t from b on ab has eccentricity
    # max(5 - t, t + 9/2), minimized at t = 1/4 with value 19/4.  The
    # discrete radius is ecc(b) = 5, attained only at b.
    rad = cx.radius(G)
    assert rad == F(19, 4)
    assert nx.radius(g, weight="length") == 5
    assert rad <= diam <= 2 * rad

    # ------------------------------------------------------------------
    # 7. Center.
    # ------------------------------------------------------------------
    # A single point interior to ab, 19/4 from a: no vertex is central.
    center = cx.center(G)
    assert center == [cx.Point("a", "b", F(19, 20))]
    (c,) = center
    assert not c.is_vertex
    assert G.offset(c) == F(19, 4)
    assert cx.eccentricity(G, c) == rad
    assert all(cx.eccentricity(G, G.point(v)) > rad for v in g)
    assert nx.center(g, weight="length") == ["b"]

    # The eccentricity along ab has its minimum exactly there, and it
    # agrees with the eccentricities of the endpoints.
    profile = cx.eccentricity_function(G, "a", "b")
    assert profile(0) == cx.eccentricity(G, G.point("a")) == diam
    assert profile(1) == cx.eccentricity(G, G.point("b")) == 5
    assert profile.minimum() == (rad, [(F(19, 20), F(19, 20))])
