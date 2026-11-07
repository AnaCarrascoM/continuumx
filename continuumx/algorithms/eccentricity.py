"""Eccentricity of a point of a continuous graph.

The eccentricity of a point (Definition 4.4) is a maximum over the whole
continuum,

    ecc(p) = max { d_𝒢(p, q) : q ∈ 𝒢 },

but it is computable because the maximum over a single edge, the eccentricity
of p relative to that edge (Definition 4.8), has a closed form.  The global
maximum is then a maximum over the m edges, as in (4.11).

Read as functions of the position of p along a fixed edge, the eccentricities
relative to the edges are the hat functions of
:mod:`continuumx.structures.piecewise` (Theorem 4.1), and their upper envelope
is the eccentricity along that edge, from which the diameter, the radius and
the center all follow.
"""

from __future__ import annotations

from typing import Hashable, List, Tuple

from continuumx.algorithms.distance import distance_to_vertex
from continuumx.classes.continuous_graph import as_continuous
from continuumx.classes.point import Point, orient
from continuumx.structures.piecewise import HatFunction, PiecewiseLinear, upper_envelope

__all__ = [
    "eccentricity",
    "eccentricity_to_edge",
    "eccentricity_function",
    "farthest_point_on_edge",
    "hat_functions",
]


def eccentricity_to_edge(
    G, p: Point, u: Hashable, v: Hashable, *, length: str = "length"
):
    """Return ecc(p, uv) = max_{q ∈ uv} d(p, q).

    Two cases, following Lemma 4.2 and Theorem 4.1 (2) of the master's thesis.

    *The point is not interior to* uv.  The farthest point of the edge is the
    one at which the two ways in, through u and through v, cost the same, so
    by (4.9)

        ecc(p, uv) = (d(p, u) + d(p, v) + ℓ(uv)) / 2.

    That point always lies on the edge: the triangle inequality applied through
    the edge itself gives |d(p, u) - d(p, v)| ≤ ℓ(uv), so no clipping is
    needed.

    *The point is interior to* uv.  The formula above no longer applies,
    because a walk from p to a point of its own edge may stay inside the edge.
    Writing D = d_G(u, v) and L = ℓ(uv), the eccentricity relative to uv is the
    valley (4.14), which can be written as

        ecc(p, uv) = min( max(θ, 1 - θ) · L,  (L + D) / 2 ).
    """
    G = as_continuous(G, length=length)
    G.check_point(p)
    a, b, _ = orient(u, v)
    ell = G.edge_length(a, b)

    if (not p.is_vertex) and (p.u, p.v) == (a, b):
        d_uv = G.vertex_distance(a, b)
        theta = p.theta
        return min(max(theta, 1 - theta) * ell, (ell + d_uv) / 2)

    return (distance_to_vertex(G, p, a) + distance_to_vertex(G, p, b) + ell) / 2


def farthest_point_on_edge(
    G, p: Point, u: Hashable, v: Hashable, *, length: str = "length"
) -> Point:
    """Return a point of the edge ``uv`` at maximum distance from ``p``.

    If ``p`` is not interior to ``uv``, it is the point of parameter (4.10)
    (Lemma 4.2); otherwise it follows from the valley (4.14).
    :func:`~continuumx.diametral_pair` uses it to find the second point of the
    pair.
    """
    G = as_continuous(G, length=length)
    G.check_point(p)
    a, b, _ = orient(u, v)
    ell = G.edge_length(a, b)
    # 0 and 1 with the numeric type of the lengths (exact for Fraction)
    zero = ell - ell
    one = zero + 1

    if (not p.is_vertex) and (p.u, p.v) == (a, b):
        # The valley (4.14), with β = (ℓ(uv) + d_G(u, v)) / (2 ℓ(uv)).
        beta = (ell + G.vertex_distance(a, b)) / (2 * ell)
        theta = p.theta
        if theta <= 1 - beta:
            return Point(a, b, min(theta + beta, one))  # guard against rounding
        if theta >= beta:
            return Point(a, b, max(theta - beta, zero))
        return Point.at_vertex(b if 1 - theta >= theta else a)

    d_a = distance_to_vertex(G, p, a)
    d_b = distance_to_vertex(G, p, b)
    mu = (d_b + ell - d_a) / (2 * ell)
    mu = min(max(mu, zero), one)  # guard against rounding
    return Point(a, b, mu)


def eccentricity(G, p: Point, *, length: str = "length"):
    """Return the eccentricity ecc(p) of a point.

    Complexity
    ----------
    O(m) once the distances between vertices from the endpoints of the edge
    containing ``p`` are available.

    Examples
    --------
    >>> import networkx as nx, continuumx as cx
    >>> G = cx.ContinuousGraph(nx.path_graph(3))       # 0 - 1 - 2, unit lengths
    >>> cx.eccentricity(G, G.point(1))                 # the middle vertex
    1.0
    >>> cx.eccentricity(G, G.point(0))
    2.0
    """
    G = as_continuous(G, length=length)
    G.check_point(p)
    return max(eccentricity_to_edge(G, p, u, v) for u, v in G.edges)


def hat_functions(
    G, u: Hashable, v: Hashable, *, length: str = "length"
) -> List[Tuple[HatFunction, Tuple]]:
    """Return the m+1 hat functions of the edge uv, with θ measured from ``u``.

    Each entry is a pair ``(hat, wz)``: ``hat`` describes θ ↦ ecc(x_θ, wz),
    and ``wz`` is the edge it comes from (the edge uv itself contributes two
    degenerate hats, and appears twice).

    Notes
    -----
    For an edge wz ≠ uv, let θ_w and θ_z be the parameters of the points of uv
    farthest from w and from z, given by (4.7):

        θ_w = (ℓ(uv) + d(v, w) - d(u, w)) / (2 ℓ(uv)),

    and likewise θ_z; both lie in [0, 1] by Lemma 4.1.  By Theorem 4.1 (1), the
    hat has slope ℓ(uv) and attains its ceiling (4.12) on the interval
    [b_wz, s_wz] = [min(θ_w, θ_z), max(θ_w, θ_z)]:

        h = (ℓ(uv) + ℓ(wz) + min(d(u, w) + d(v, z), d(u, z) + d(v, w))) / 2.

    For the edge uv itself, the eccentricity relative to uv is the valley
    (4.14), the upper envelope of the two degenerate hats of ceiling β ℓ(uv)
    with (b, s) = (0, 1 - β) and (β, 1), where β = (ℓ(uv) + d(u, v)) / (2 ℓ(uv)).
    Representing it this way keeps a single code path downstream.
    """
    G = as_continuous(G, length=length)
    a, b = u, v  # θ is measured from u, as in (4.1)
    L = G.edge_length(a, b)
    # 0 and 1 with the numeric type of the lengths (exact for Fraction)
    zero = L - L
    one = zero + 1
    d_a = G.distances_from(a)
    d_b = G.distances_from(b)

    hats: List[Tuple[HatFunction, Tuple]] = []

    for w, z, ell in G.iter_edges_with_length():
        if {w, z} == {a, b}:
            D = d_a[b]
            ceiling = (L + D) / 2
            theta_lo = (L - D) / (2 * L)
            theta_hi = (L + D) / (2 * L)
            hats.append((HatFunction(zero, theta_lo, ceiling, L), (a, b)))
            hats.append((HatFunction(theta_hi, one, ceiling, L), (a, b)))
            continue
        theta_w = (L + d_b[w] - d_a[w]) / (2 * L)
        theta_z = (L + d_b[z] - d_a[z]) / (2 * L)
        # in [0, 1] by Lemma 4.1; guard against rounding
        theta_w = min(max(theta_w, zero), one)
        theta_z = min(max(theta_z, zero), one)
        # then the min in (4.12) is d(v, w) + d(u, z)
        if theta_w <= theta_z:
            lo, hi = theta_w, theta_z
            cross = d_b[w] + d_a[z]
        else:
            lo, hi = theta_z, theta_w
            cross = d_b[z] + d_a[w]
        ceiling = (L + ell + cross) / 2
        hats.append((HatFunction(lo, hi, ceiling, L), (w, z)))

    return hats


def eccentricity_function(
    G, u: Hashable, v: Hashable, *, length: str = "length"
) -> PiecewiseLinear:
    """Return θ ↦ ecc(x_θ) along the edge ``uv``, with θ measured from ``u``.

    The result is the upper envelope of the m+1 hat functions of the
    edge; the local radius and the local center of the edge are read off it.

    Complexity
    ----------
    O(m²) once the distances between vertices are available: m+1 insertions into
    the envelope, each O(m) (see :func:`~continuumx.upper_envelope`).

    Examples
    --------
    >>> import networkx as nx, continuumx as cx
    >>> G = cx.ContinuousGraph(nx.path_graph(3))
    >>> f = cx.eccentricity_function(G, 0, 1)
    >>> f(0.0), f(1.0)                      # eccentricities of vertices 0 and 1
    (2.0, 1.0)
    """
    G = as_continuous(G, length=length)
    return upper_envelope(hat for hat, _ in hat_functions(G, u, v))
