"""The :class:`ContinuousGraph` class.

A continuous graph is determined by the edge-weighted graph (G, ℓ) that
defines it (Definition 4.1), so no new combinatorial data is needed.
Accordingly, :class:`ContinuousGraph` is not a subclass of
:class:`networkx.Graph` but a view over one, which it neither copies nor
modifies.
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
    """The continuous graph 𝒢 defined by (G, ℓ) (Definition 4.1).

    Parameters
    ----------
    graph : networkx.Graph
        A simple, undirected, connected graph with at least one vertex.
    length : str, optional
        Name of the edge attribute holding ℓ.  Defaults to ``"length"``.
        Edges lacking the attribute are given length ``1``.
    tol : float, optional
        Tolerance used when comparing lengths and parameters.  Defaults to
        ``1e-9``; set it to ``0`` for exact arithmetic with
        :class:`fractions.Fraction` lengths.

    Attributes
    ----------
    graph : networkx.Graph
        The underlying combinatorial graph.  Treat it as read-only: the view
        caches shortest-path distances computed from it, and mutating it
        invalidates them silently.

    Notes
    -----
    Directed graphs, multigraphs, self-loops, and edges whose length is not
    positive and finite are rejected with an explicit error.

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
            raise UnsupportedGraphError("self-loops are not supported in this version")
        if not nx.is_connected(graph):
            raise NotConnectedError(
                "the graph must be connected for the continuous distance to be "
                "finite between every pair of points"
            )

        self._graph = graph
        self._length_attr = length
        self._tol = tol
        self._dist_cache: Dict[Hashable, Dict[Hashable, Any]] = {}

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
                    "strictly positive"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise InvalidLengthError(
                    f"edge ({u!r}, {v!r}) has non-finite length {value!r}"
                )
            a, b, _ = orient(u, v)
            self._edges.append((a, b))
            self._lengths[(a, b)] = value

        if self._graph.number_of_edges() and not saw_attribute:
            # If the graph was built with weight=, the NetworkX default, no
            # edge carries the length attribute, and falling back to unit
            # lengths would silently give wrong values.
            has_weight = any("weight" in d for _, _, d in self._graph.edges(data=True))
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
        """The underlying :class:`networkx.Graph`."""
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
        """Return the total length ℓ(𝒢) = Σ_{uv∈E} ℓ(uv)."""
        if not self._lengths:
            return 0.0
        total = None
        for value in self._lengths.values():
            total = value if total is None else total + value
        return total

    def offset(self, p: Point):
        """Return the offset λ = θ ℓ(uv), measured from the first endpoint ``p.u``.

        A vertex is stored as ``(w, w, 0)``, with no edge, so its offset is ``0``.
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
            Parameter θ ∈ [0, 1] of the point along ``uv``, measured from
            ``u``.
        offset : float, optional
            Distance λ = ℓ(u, p) from ``u`` along ``uv``, in the units of the
            edge lengths; converted to θ = λ / ℓ(uv).  At most one of
            ``theta`` and ``offset`` may be given; if neither is, the point is
            the vertex ``u``.

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
            # 0 with the numeric type of the lengths (exact for Fraction)
            zero = length - length
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
            raise InvalidPointError(f"({p.u!r}, {p.v!r}) is not an edge of the graph")
        return p

    def same_point(self, p: Point, q: Point) -> bool:
        """Whether ``p`` and ``q`` are the same point of 𝒢.

        ``p == q`` already recognizes different descriptions of the same point,
        since points are stored in canonical form (see
        :class:`~continuumx.Point`).  This method also accepts points whose
        distance is at most the graph tolerance, to allow for rounding errors
        with ``float`` lengths.
        """
        from continuumx.algorithms.distance import distance

        if p == q:
            return True
        return distance(self, p, q) <= self._tol

    # ------------------------------------------------------------------
    # Distances between vertices
    # ------------------------------------------------------------------

    def distances_from(self, source: Hashable) -> Dict[Hashable, Any]:
        """Return the distances in G from ``source`` to every vertex.

        They are computed with Dijkstra's algorithm the first time and stored,
        so later calls from the same source do not compute them again.  When
        only a few sources are needed, this is cheaper than :meth:`precompute`,
        which computes and stores the distances from every vertex, using O(n²)
        memory.
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
        """Compute the distances between all pairs of vertices and return ``self``.

        The global algorithms (:func:`~continuumx.diameter`,
        :func:`~continuumx.diametral_pair`, :func:`~continuumx.radius`,
        :func:`~continuumx.center`) call this themselves.  It is public so that
        timings can separate the shortest-path stage, which takes A(n, m) =
        O(n m log n) time with Dijkstra's algorithm, from the continuous stage.
        """
        for node in self._graph.nodes():
            self.distances_from(node)
        return self

    def _weight_function(self):
        """Return the edge-weight function for NetworkX's Dijkstra: the length
        attribute of each edge, or 1 if it is missing."""
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
    """Return ``graph`` as a :class:`ContinuousGraph`.

    A :class:`ContinuousGraph` is returned unchanged, so that the distances it
    has already computed are reused; a :class:`networkx.Graph` is wrapped in a
    new one.  Every public function of the library calls this on its input.
    """
    if isinstance(graph, ContinuousGraph):
        return graph
    return ContinuousGraph(graph, length=length, tol=tol)
