"""The :class:`ContinuousGraph` class.

A continuous graph adds no combinatorial data to a graph: it is a
reinterpretation of a pair (G, ℓ), where G is a connected graph and ℓ assigns
a length to each edge.  Accordingly, :class:`ContinuousGraph` is not a subclass
of :class:`networkx.Graph` but a view over one.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Hashable, Iterator, List, Optional, Tuple

import networkx as nx

from continuumx.classes.point import Point, orient
from continuumx.exceptions import (
    InvalidLengthError,
    InvalidPointError,
    NotConnectedError,
    UnsupportedGraphError,
)

__all__ = ["ContinuousGraph"]

#: Default tolerance used for comparisons of lengths and parameters.
DEFAULT_TOL = 1e-9


class ContinuousGraph:
    """The continuous graph 𝒢 defined by (G, ℓ).

    Parameters
    ----------
    graph : networkx.Graph
        A simple, undirected, connected graph with at least one vertex.
    length : str, optional
        Name of the edge attribute holding ℓ.  Defaults to ``"length"``.
        Edges lacking the attribute are given length ``1``.
    tol : float, optional
        Tolerance used when comparing lengths and parameters.  Set it to ``0``
        for exact arithmetic (see *Exact arithmetic* below).

    Attributes
    ----------
    graph : networkx.Graph
        The underlying combinatorial graph.  Treat it as read-only: the view
        caches shortest-path distances computed from it, and mutating it
        invalidates them silently.

    Notes
    -----
    .. _zero-length:

    **Supported inputs.**  Version 0.1 deliberately restricts the input class
    and rejects everything else with an explicit error rather than failing
    later or, worse, returning a wrong number:

    * *Directed graphs and multigraphs* are rejected.  Multigraphs are a
      genuine loss — two parallel edges of different lengths form a perfectly
      good continuous graph, indeed a continuous C₂ — but the triple
      (u, v, θ) cannot name one of two parallel edges.  Adding them later
      means adding an edge key to :class:`~continuumx.Point`, not rewriting
      the algorithms.
    * *Self-loops* are rejected for the same reason (a loop is a circle
      attached at a vertex, and both its "endpoints" are the same vertex).
    * *Zero-length edges* are rejected.  The theory admits them, but
      θ = λ / ℓ is then undefined and every formula acquires a degenerate
      branch, in exchange for no expressive power: contracting a zero-length
      edge yields an isometric continuous graph.

    **Exact arithmetic.**  Every formula in the library uses only the four
    arithmetic operations, ``min`` and ``max``, with no square roots or other
    functions.  Consequently, if the edge lengths are
    :class:`fractions.Fraction` objects, all computed parameters are exact
    rationals and ``tol=0`` is safe.  This matters most for the center, whose
    combinatorial structure is decided by equality tests between breakpoints.

    Examples
    --------
    >>> import networkx as nx, continuumx as cx
    >>> g = nx.path_graph(3)
    >>> G = cx.ContinuousGraph(g)          # unit lengths by default
    >>> G.total_length()
    2.0
    >>> G.vertex_distance(0, 2)
    2.0
    """

    __slots__ = ("_graph", "_length_attr", "_tol", "_dist_cache", "_edges", "_lengths")

    def __init__(
        self,
        graph: nx.Graph,
        length: str = "length",
        tol: float = DEFAULT_TOL,
    ) -> None:
        if graph is None:
            raise UnsupportedGraphError("a graph is required")
        if graph.is_directed():
            raise UnsupportedGraphError(
                "continuous graphs are undirected; convert with "
                "`graph.to_undirected()` first"
            )
        if graph.is_multigraph():
            raise UnsupportedGraphError(
                "multigraphs are not supported in this version: a point "
                "(u, v, theta) cannot name one of several parallel edges"
            )
        if graph.number_of_nodes() == 0:
            raise UnsupportedGraphError("the graph must have at least one vertex")
        if nx.number_of_selfloops(graph) > 0:
            raise UnsupportedGraphError(
                "self-loops are not supported in this version"
            )
        if not nx.is_connected(graph):
            raise NotConnectedError(
                "the graph must be connected for the continuous distance to be "
                "finite between every pair of points"
            )

        self._graph = graph
        self._length_attr = length
        self._tol = tol
        self._dist_cache: Dict[Hashable, Dict[Hashable, float]] = {}

        self._edges: List[Tuple[Hashable, Hashable]] = []
        self._lengths: Dict[Tuple[Hashable, Hashable], Any] = {}
        self._collect_lengths()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _collect_lengths(self) -> None:
        attr = self._length_attr
        saw_attribute = False
        for u, v, data in self._graph.edges(data=True):
            if attr in data:
                saw_attribute = True
                value = data[attr]
            else:
                value = 1.0
            if value <= 0:
                raise InvalidLengthError(
                    f"edge ({u!r}, {v!r}) has length {value!r}; lengths must be "
                    "strictly positive (see the note on zero-length edges)"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise InvalidLengthError(
                    f"edge ({u!r}, {v!r}) has non-finite length {value!r}"
                )
            a, b, _ = orient(u, v)
            self._edges.append((a, b))
            self._lengths[(a, b)] = value

        if self._graph.number_of_edges() and not saw_attribute:
            # A very common mistake: the graph was built with `weight=`, the
            # networkx default, but the library looks for `length`.  Silently
            # falling back to unit lengths would return plausible wrong numbers.
            has_weight = any(
                "weight" in d for _, _, d in self._graph.edges(data=True)
            )
            if has_weight and self._length_attr != "weight":
                raise InvalidLengthError(
                    f"no edge carries the attribute {self._length_attr!r}, but "
                    "some carry 'weight'; pass length='weight' if that is the "
                    "attribute holding the edge lengths"
                )

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.Graph:
        """The underlying :class:`networkx.Graph` (read-only by contract)."""
        return self._graph

    @property
    def length_attr(self) -> str:
        """Name of the edge attribute holding the lengths."""
        return self._length_attr

    @property
    def tol(self) -> float:
        """Tolerance used in comparisons."""
        return self._tol

    @property
    def nodes(self) -> List[Hashable]:
        """List of vertices."""
        return list(self._graph.nodes())

    @property
    def edges(self) -> List[Tuple[Hashable, Hashable]]:
        """List of edges, each with its endpoints in canonical order."""
        return list(self._edges)

    def number_of_nodes(self) -> int:
        """Number of vertices, n."""
        return self._graph.number_of_nodes()

    def number_of_edges(self) -> int:
        """Number of edges, m."""
        return self._graph.number_of_edges()

    def __len__(self) -> int:
        return self.number_of_nodes()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<ContinuousGraph with {self.number_of_nodes()} vertices, "
            f"{self.number_of_edges()} edges, total length {self.total_length()}>"
        )

    # ------------------------------------------------------------------
    # Lengths
    # ------------------------------------------------------------------

    def edge_length(self, u: Hashable, v: Hashable):
        """Return ℓ(uv).

        Raises
        ------
        ValueError
            If ``uv`` is not an edge of the graph.
        """
        a, b, _ = orient(u, v)
        try:
            return self._lengths[(a, b)]
        except KeyError:
            raise ValueError(f"({u!r}, {v!r}) is not an edge of the graph") from None

    def total_length(self):
        """Return ℓ(E) = Σ_{e∈E} ℓ(e)."""
        if not self._lengths:
            return 0.0
        total = None
        for value in self._lengths.values():
            total = value if total is None else total + value
        return total

    def offset(self, p: Point):
        """Return the absolute offset λ = θ ℓ(uv).

        For a vertex the offset is ``0``.
        """
        if p.is_vertex:
            return 0.0
        return p.theta * self.edge_length(p.u, p.v)

    # ------------------------------------------------------------------
    # Points
    # ------------------------------------------------------------------

    def point(
        self,
        u: Hashable,
        v: Optional[Hashable] = None,
        theta: Optional[float] = None,
        offset: Optional[float] = None,
    ) -> Point:
        """Build a validated point of this graph.

        Parameters
        ----------
        u, v : hashable
            Endpoints of the edge.  Omit ``v`` (or pass ``v == u``) for the
            vertex ``u``.
        theta : float, optional
            Normalized position along ``uv``, measured from ``u``.
        offset : float, optional
            Absolute distance from ``u`` along ``uv``; converted to ``theta``.
            Exactly one of ``theta`` and ``offset`` may be given.

        Examples
        --------
        >>> import networkx as nx, continuumx as cx
        >>> G = cx.ContinuousGraph(nx.path_graph(3))
        >>> G.point(0, 1, theta=0.5)
        Point(0, 1, 0.5)
        >>> G.point(2)
        Point(2)
        """
        if v is None or v == u:
            if u not in self._graph:
                raise InvalidPointError(f"{u!r} is not a vertex of the graph")
            return Point.at_vertex(u)
        if theta is not None and offset is not None:
            raise TypeError("give either theta or offset, not both")
        length = self.edge_length(u, v)
        if offset is not None:
            if offset < -self._tol or offset > length + self._tol:
                raise InvalidPointError(
                    f"offset {offset!r} is outside [0, {length!r}], the length "
                    f"of edge ({u!r}, {v!r})"
                )
            zero = length - length      # numeric type of the lengths (exact for Fraction)
            theta = min(max(offset / length, zero), zero + 1)
        elif theta is None:
            theta = length - length
        return Point(u, v, theta)

    def check_point(self, p: Point) -> Point:
        """Validate that ``p`` is a point of this graph and return it."""
        if not isinstance(p, Point):
            raise InvalidPointError(f"expected a Point, got {type(p).__name__}")
        if p.is_vertex:
            if p.u not in self._graph:
                raise InvalidPointError(f"{p.u!r} is not a vertex of the graph")
        elif not self._graph.has_edge(p.u, p.v):
            raise InvalidPointError(
                f"({p.u!r}, {p.v!r}) is not an edge of the graph"
            )
        return p

    def same_point(self, p: Point, q: Point) -> bool:
        """Whether ``p`` and ``q`` denote the same point of 𝒢.

        Structural equality already identifies the triples that describe the
        same point (see :class:`~continuumx.Point`); this method additionally
        absorbs floating-point noise, using the graph tolerance.
        """
        from continuumx.algorithms.distance import distance

        if p == q:
            return True
        return distance(self, p, q) <= self._tol

    # ------------------------------------------------------------------
    # Discrete distances
    # ------------------------------------------------------------------

    def distances_from(self, source: Hashable) -> Dict[Hashable, float]:
        """Single-source shortest-path distances in G, cached.

        The result is memoized, so repeated queries from the same source are
        free.  Prefer this to :meth:`precompute` when only a few sources are
        needed: computing the full matrix eagerly costs O(n²) memory
        even for a single distance query.
        """
        cached = self._dist_cache.get(source)
        if cached is None:
            if source not in self._graph:
                raise ValueError(f"{source!r} is not a vertex of the graph")
            cached = nx.single_source_dijkstra_path_length(
                self._graph, source, weight=self._weight_function()
            )
            self._dist_cache[source] = cached
        return cached

    def vertex_distance(self, u: Hashable, v: Hashable):
        """Return the shortest-path distance d_G(u, v) between vertices."""
        return self.distances_from(u)[v]

    def precompute(self) -> "ContinuousGraph":
        """Fill the all-pairs distance cache and return ``self``.

        Global algorithms (:func:`~continuumx.diameter`,
        :func:`~continuumx.radius`, :func:`~continuumx.center`) call this
        themselves; it is exposed so that timings can separate the
        O(n m log n) shortest-path stage from the continuous stage.
        """
        for node in self._graph.nodes():
            self.distances_from(node)
        return self

    def _weight_function(self):
        attr = self._length_attr
        def weight(_u, _v, data):
            return data.get(attr, 1.0)
        return weight

    # ------------------------------------------------------------------
    # Iteration helpers used by the algorithms
    # ------------------------------------------------------------------

    def iter_edges_with_length(self) -> Iterator[Tuple[Hashable, Hashable, Any]]:
        """Yield ``(u, v, length)`` for every edge, endpoints canonically ordered."""
        for u, v in self._edges:
            yield u, v, self._lengths[(u, v)]


def as_continuous(graph, length: str = "length", tol: float = DEFAULT_TOL):
    """Coerce ``graph`` to a :class:`ContinuousGraph`.

    Accepts a :class:`ContinuousGraph` (returned unchanged, so that its caches
    are reused) or a :class:`networkx.Graph` (wrapped).  Every public algorithm
    of the library funnels its input through this function.
    """
    if isinstance(graph, ContinuousGraph):
        return graph
    return ContinuousGraph(graph, length=length, tol=tol)
