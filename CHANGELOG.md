# Changelog

All notable changes to this project are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows [Semantic Versioning](https://semver.org/).

## [0.1.0]

First version, described in Chapter 5 of the master's thesis *Computing
Distances in Combinatorial and Continuous Graphs*.

### Added

- `ContinuousGraph`, a view over a NetworkX graph with edge lengths, and the
  classes `Point` and `Segment`.
- Distance between two points (`distance`, `distance_to_vertex`).
- Eccentricity of a point, relative to an edge and along an edge
  (`eccentricity`, `eccentricity_to_edge`, `eccentricity_function`,
  `farthest_point_on_edge`).
- Diameter and a diametral pair (`diameter`, `diametral_pair`).
- Local radius and local center of an edge, and radius and center of the
  graph (`local_radius`, `local_center`, `radius`, `center`).
- Closed forms for trees and cycles (`tree_*`, `cycle_*`).
- Exact computations with `fractions.Fraction` edge lengths.
- Test suite and an example notebook on the emergency network.

[0.1.0]: https://github.com/AnaCarrascoM/continuumx/releases/tag/v0.1.0
