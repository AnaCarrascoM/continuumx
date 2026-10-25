"""Algorithms on continuous graphs."""

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

__all__ = [
    "distance",
    "distance_to_vertex",
    "eccentricity",
    "eccentricity_to_edge",
    "eccentricity_function",
    "farthest_point_on_edge",
    "hat_functions",
    "diameter",
    "diametral_pair",
    "radius",
    "local_radius",
    "center",
    "local_center",
    "is_tree",
    "is_cycle",
    "tree_diameter",
    "tree_radius",
    "tree_center",
    "cycle_diameter",
    "cycle_radius",
    "cycle_center",
]
