"""Exception hierarchy of :mod:`continuumx`.

Invalid inputs (graphs outside the supported class, inadmissible lengths and
triples that are not points of the graph) raise subclasses of
:class:`ContinuumXError`, so that user code can catch them uniformly::

    try:
        G = cx.ContinuousGraph(g)
    except cx.ContinuumXError:
        ...

Other misuses, such as asking for the length of a pair that is not an edge,
raise the standard :class:`ValueError` or :class:`TypeError`.
"""

__all__ = [
    "ContinuumXError",
    "NotConnectedError",
    "InvalidLengthError",
    "UnsupportedGraphError",
    "InvalidPointError",
]


class ContinuumXError(Exception):
    """Base class for every exception raised by :mod:`continuumx`."""


class NotConnectedError(ContinuumXError):
    """The underlying combinatorial graph is not connected.

    A continuous graph is defined from a *connected* graph, so that the
    shortest-path distance is finite for every pair of points.
    """


class InvalidLengthError(ContinuumXError):
    """An edge carries a length that is not admissible.

    Lengths must be strictly positive and finite.
    """


class UnsupportedGraphError(ContinuumXError):
    """The input graph is outside the class supported by this version.

    Version 0.1 supports simple, undirected, connected graphs with at least
    one vertex only: directed graphs, multigraphs, self-loops and graphs
    without vertices are rejected explicitly rather than handled silently.
    """


class InvalidPointError(ContinuumXError):
    """A triple does not describe a point of the continuous graph."""
