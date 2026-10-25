"""Eccentricity of a point of a continuous graph.

The eccentricity of a point is a maximum over the whole continuum,

    ecc(p) = max { d_𝒢(p, q) : q ∈ 𝒢 },

but it is computable because the maximum over a single edge — the *remoteness*
of p relative to that edge — has a closed form.  The global maximum is then a
maximum over the m edges.

Reading the eccentricity as a function of the position of p along a fixed edge
gives the hat functions of
:mod:`continuumx.structures.piecewise`, from which the diameter, the radius and
the centre all follow.
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


def eccentricity_to_edge(G, p: Point, u: Hashable, v: Hashable, *,
                         length: str = "length"):
    """Return ecc(p, uv) = max_{q ∈ uv} d(p, q).

    Two cases, following Theorem 4.x of the thesis.

    *The point is not interior to* uv.  The farthest point of the edge is the
    one at which the two ways in — through u and through v — cost the same, so

        ecc(p, uv) = (d(p, u) + d(p, v) + ℓ(uv)) / 2.

    That point always lies on the edge: the triangle inequality applied through
    the edge itself gives |d(p, u) - d(p, v)| ≤ ℓ(uv), so no clipping is
    needed.

    *The point is interior to* uv.  The formula above is then wrong, because a
    path from p to a point of its own edge may simply stay inside the edge.
    Writing D = d_G(u, v) and L = ℓ(uv), the remoteness becomes the "valley"

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


def farthest_point_on_edge(G, p: Point, u: Hashable, v: Hashable, *,
                           length: str = "length") -> Point:
    """Return a point of the edge ``uv`` at maximum distance from ``p``.

    The value of the maximum is :func:`eccentricity_to_edge`; this function
    returns a point attaining it, which is what witnesses a diametral pair.
    """
    G = as_continuous(G, length=length)
    G.check_point(p)
    a, b, _ = orient(u, v)
    ell = G.edge_length(a, b)
    zero = ell - ell      # 0 and 1 of the same numeric type as the lengths,
    one = zero + 1        # so that Fraction inputs never meet a float

    if (not p.is_vertex) and (p.u, p.v) == (a, b):
        D = G.vertex_distance(a, b)
        lam = p.theta * ell
        if lam <= (ell - D) / 2:
            return Point(a, b, (lam + (ell + D) / 2) / ell)
        if lam >= (ell + D) / 2:
            return Point(a, b, (lam - (ell + D) / 2) / ell)
        return Point.at_vertex(b if ell - lam >= lam else a)

    d_a = distance_to_vertex(G, p, a)
    d_b = distance_to_vertex(G, p, b)
    mu = (d_b + ell - d_a) / (2 * ell)
    mu = min(max(mu, zero), one)
    return Point(a, b, mu)


def eccentricity(G, p: Point, *, length: str = "length"):
    """Return the continuous eccentricity ecc(p).

    Complexity
    ----------
    O(m) once the shortest-path distances from the endpoints of the
    edge containing ``p`` are cached.

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
    return max(
        eccentricity_to_edge(G, p, u, v) for u, v in G.edges
    )


def hat_functions(G, u: Hashable, v: Hashable, *,
                  length: str = "length") -> List[Tuple[HatFunction, Tuple]]:
    """Return the m+1 hat functions of the edge uv.

    Each entry is a pair ``(hat, witness)`` where ``hat`` describes
    θ ↦ ecc(x_θ, e') for one edge e', and ``witness`` is that edge (the edge
    uv itself contributes two degenerate hats, and is reported twice).

    Notes
    -----
    For e' = v_k v_l ≠ uv, write

        θ_k = (ℓ(uv) + d(v, v_k) - d(u, v_k)) / (2 ℓ(uv)),

    the position of the point of uv farthest from v_k, and likewise θ_l.  Both
    lie in [0, 1] by the triangle inequality.  The hat has plateau
    [min(θ_k, θ_l), max(θ_k, θ_l)], slope ℓ(uv), and ceiling

        h = (ℓ(uv) + ℓ(e') + min(d(u, v_k) + d(v, v_l), d(u, v_l) + d(v, v_k))) / 2.

    For e' = uv, the remoteness is the valley of :func:`eccentricity_to_edge`,
    which is the upper envelope of the two degenerate hats with ceiling
    (ℓ(uv) + d_G(u, v)) / 2 and plateaus [0, θ_v] and [θ_u, 1].  Modelling it
    this way keeps a single code path downstream.
    """
    G = as_continuous(G, length=length)
    a, b, _ = orient(u, v)
    L = G.edge_length(a, b)
    zero = L - L          # 0 and 1 of the same numeric type as the lengths,
    one = zero + 1        # so that Fraction inputs never meet a float
    d_a = G.distances_from(a)
    d_b = G.distances_from(b)

    hats: List[Tuple[HatFunction, Tuple]] = []

    for k, l, ell in G.iter_edges_with_length():
        if (k, l) == (a, b):
            D = d_a[b]
            ceiling = (L + D) / 2
            theta_lo = (L - D) / (2 * L)
            theta_hi = (L + D) / (2 * L)
            hats.append((HatFunction(zero, theta_lo, ceiling, L), (a, b)))
            hats.append((HatFunction(theta_hi, one, ceiling, L), (a, b)))
            continue
        theta_k = (L + d_b[k] - d_a[k]) / (2 * L)
        theta_l = (L + d_b[l] - d_a[l]) / (2 * L)
        theta_k = min(max(theta_k, zero), one)  # guard against rounding only
        theta_l = min(max(theta_l, zero), one)
        if theta_k <= theta_l:
            lo, hi = theta_k, theta_l
            cross = d_b[k] + d_a[l]
        else:
            lo, hi = theta_l, theta_k
            cross = d_b[l] + d_a[k]
        ceiling = (L + ell + cross) / 2
        hats.append((HatFunction(lo, hi, ceiling, L), (k, l)))

    return hats


def eccentricity_function(G, u: Hashable, v: Hashable, *,
                          length: str = "length") -> PiecewiseLinear:
    """Return θ ↦ ecc(x_θ) along the edge ``uv``.

    The result is the upper envelope of the m+1 hat functions of the
    edge.  It is the object the local radius and local centre are read off, and
    it is also what to plot when illustrating the eccentricity profile of an
    edge.

    Complexity
    ----------
    O(m log m) once the discrete distances are available.

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
