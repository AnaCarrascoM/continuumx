"""Drawing continuous graphs with matplotlib.

The point of a drawing here is not the graph — ``networkx`` draws graphs — but
the *sets of points* the library computes: a centre that is a union of
sub-paths, a diametral pair inside two edges.  These have no representation in
``networkx``, so :func:`draw` layers them on top of a standard drawing.

``matplotlib`` is an optional dependency; install it with
``pip install "continuumx[drawing]"``.
"""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, Optional, Sequence

import networkx as nx

from continuumx.classes.continuous_graph import as_continuous
from continuumx.classes.point import Point, Segment

__all__ = ["draw", "point_position"]


def point_position(G, p: Point, pos: Dict[Hashable, Sequence[float]]):
    """Return the plane coordinates of a point, interpolating along its edge."""
    if p.is_vertex:
        return tuple(pos[p.u])
    a, b = pos[p.u], pos[p.v]
    t = p.theta
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def draw(
    G,
    pos: Optional[Dict[Hashable, Sequence[float]]] = None,
    *,
    points: Iterable[Point] = (),
    segments: Iterable[Segment] = (),
    ax=None,
    pos_attr: str = "pos",
    length: str = "length",
    node_size: int = 90,
    point_color: str = "#c1440e",
    segment_color: str = "#c1440e",
    segment_width: float = 4.0,
    with_labels: bool = True,
    **kwargs,
):
    """Draw a continuous graph, highlighting points and segments.

    Parameters
    ----------
    G : ContinuousGraph or networkx.Graph
    pos : dict, optional
        Vertex coordinates.  Falls back to the node attribute ``pos_attr``, and
        then to ``networkx.kamada_kawai_layout`` weighted by the edge lengths,
        so that the picture reflects the metric.
    points : iterable of Point
        Marked with a dot — e.g. a diametral pair.
    segments : iterable of Segment
        Drawn as thick sub-paths — e.g. the continuous centre.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "drawing requires matplotlib; install continuumx[drawing]"
        ) from exc

    G = as_continuous(G, length=length)
    if pos is None:
        pos = nx.get_node_attributes(G.graph, pos_attr)
    if not pos:
        pos = nx.kamada_kawai_layout(G.graph, weight=G.length_attr)

    if ax is None:
        _, ax = plt.subplots()

    nx.draw_networkx(
        G.graph,
        pos=pos,
        ax=ax,
        node_size=node_size,
        with_labels=with_labels,
        **kwargs,
    )

    for seg in segments:
        p, q = seg.endpoints()
        (x0, y0), (x1, y1) = point_position(G, p, pos), point_position(G, q, pos)
        if seg.is_point:
            ax.plot([x0], [y0], "o", color=segment_color, markersize=segment_width * 2)
        else:
            ax.plot([x0, x1], [y0, y1], "-", color=segment_color,
                    linewidth=segment_width, solid_capstyle="round", zorder=3)

    for p in points:
        x, y = point_position(G, p, pos)
        ax.plot([x], [y], "o", color=point_color, markersize=9, zorder=4)

    ax.set_axis_off()
    return ax
