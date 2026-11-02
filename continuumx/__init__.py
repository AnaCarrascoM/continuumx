"""continuumx: continuous graphs on top of NetworkX.

In a continuous graph, each edge of a connected graph is identified with a
segment whose length is given by its weight, and distances are measured
between any two points of the graph, not only between its vertices.  This
package implements the constructions and algorithms of the master's thesis
*Computing Distances in Combinatorial and Continuous Graphs* (A. Carrasco
Martín, Universidad de Sevilla, 2026).

Quick start
-----------
>>> import networkx as nx
>>> import continuumx as cx
>>> g = nx.Graph()
>>> g.add_edge("a", "b", length=5.0)
>>> g.add_edge("a", "c", length=1.0)
>>> g.add_edge("c", "b", length=1.0)
>>> G = cx.ContinuousGraph(g)
>>> p = G.point("a", "b", offset=0.1)
>>> q = G.point("a", "b", offset=4.9)
>>> round(cx.distance(G, p, q), 10)
2.2
>>> cx.diameter(G)
3.5
>>> cx.radius(G)                 # a cycle: every point has the same eccentricity
3.5
>>> cx.center(G)                 # ... so the center is the whole graph
[Segment('a', 'b', 0.0, 1.0), Segment('a', 'c', 0.0, 1.0), Segment('b', 'c', 0.0, 1.0)]

Naming
------
The package namespace already says "continuous", so parameters keep the names
they have in ``networkx``: :func:`diameter`, :func:`radius`, :func:`center`,
:func:`eccentricity`.  Comparing the two settings is then a matter of reading
``nx.diameter(g)`` against ``cx.diameter(g)``.
"""

from continuumx.classes.continuous_graph import ContinuousGraph, as_continuous
from continuumx.classes.point import Point, Segment
from continuumx.structures.piecewise import HatFunction, PiecewiseLinear, upper_envelope
from continuumx.algorithms.distance import distance, distance_to_vertex
from continuumx.algorithms.eccentricity import (
    eccentricity,
    eccentricity_function,
    eccentricity_to_edge,
    farthest_point_on_edge,
    hat_functions,
)
from continuumx.algorithms.diameter import diameter, diametral_pair
from continuumx.algorithms.radius import local_radius, radius
from continuumx.algorithms.center import center, local_center
from continuumx.algorithms.families import (
    cycle_center,
    cycle_diameter,
    cycle_radius,
    is_cycle,
    is_tree,
    tree_center,
    tree_diameter,
    tree_radius,
)
from continuumx.exceptions import (
    ContinuumXError,
    InvalidLengthError,
    InvalidPointError,
    NotConnectedError,
    UnsupportedGraphError,
)

__version__ = "0.1.0"

__all__ = [
    # classes
    "ContinuousGraph",
    "Point",
    "Segment",
    "as_continuous",
    # piecewise-linear toolkit
    "HatFunction",
    "PiecewiseLinear",
    "upper_envelope",
    # distances
    "distance",
    "distance_to_vertex",
    # eccentricity
    "eccentricity",
    "eccentricity_to_edge",
    "eccentricity_function",
    "farthest_point_on_edge",
    "hat_functions",
    # parameters
    "diameter",
    "diametral_pair",
    "radius",
    "local_radius",
    "center",
    "local_center",
    # families
    "is_tree",
    "is_cycle",
    "tree_diameter",
    "tree_radius",
    "tree_center",
    "cycle_diameter",
    "cycle_radius",
    "cycle_center",
    # exceptions
    "ContinuumXError",
    "NotConnectedError",
    "InvalidLengthError",
    "UnsupportedGraphError",
    "InvalidPointError",
    "__version__",
]
