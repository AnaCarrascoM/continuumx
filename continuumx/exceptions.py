"""Exception hierarchy of :mod:`continuumx`.

All exceptions raised by the library derive from :class:`ContinuumXError`, so
that user code can catch them uniformly::

    try:
        G = cx.ContinuousGraph(g)
    except cx.ContinuumXError:
        ...
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

    Lengths must be strictly positive and finite.  See
    :ref:`the note on zero-length edges <zero-length>` in
    :class:`~continuumx.classes.continuous_graph.ContinuousGraph`.
    """


class UnsupportedGraphError(ContinuumXError):
    """The input graph is outside the class supported by this version.

    Version 0.1 supports simple, undirected, connected graphs only:
    directed graphs, multigraphs and self-loops are rejected explicitly
    rather than handled silently.
    """


class InvalidPointError(ContinuumXError):
    """A triple does not describe a point of the continuous graph."""
