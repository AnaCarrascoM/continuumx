# continuumx

[![tests](https://github.com/AnaCarrascoM/continuumx/actions/workflows/tests.yml/badge.svg)](https://github.com/AnaCarrascoM/continuumx/actions/workflows/tests.yml)

`continuumx` is a Python library for **continuous graphs**, built on top of
[NetworkX](https://networkx.org/). In a combinatorial graph, distances are
measured only between vertices. In a continuous graph, each edge is replaced by
a segment of its length, and the segments are glued at their endpoints, so
distances are measured between any two points of the graph, including points
inside the edges.

It accompanies the master's thesis *Computing Distances in Combinatorial and
Continuous Graphs* (A. Carrasco Martín, Universidad de Sevilla, 2026), whose
Chapter 5 describes it. It is a first version: for arbitrary connected graphs
with positive edge lengths, it computes the distance between two points, the
eccentricity of a point, and the diameter, the radius and the center of a
continuous graph.

## Installation

`continuumx` requires Python 3.10 or later and NetworkX 3.0 or later. Version
0.1.0, the one described in the master's thesis, can be installed directly from
the repository:

```bash
pip install git+https://github.com/AnaCarrascoM/continuumx@v0.1.0
```

## Quick start

```python
import networkx as nx
import continuumx as cx

g = nx.Graph()
g.add_edge("a", "b", length=5.0)
g.add_edge("a", "c", length=1.0)
g.add_edge("c", "b", length=1.0)

G = cx.ContinuousGraph(g)

p = G.point("a", "b", offset=0.1)   # 0.1 from a, along edge ab
q = G.point("a", "b", offset=4.9)

cx.distance(G, p, q)                # 2.2
cx.diameter(G)                      # 3.5, half the perimeter: G is a cycle
cx.radius(G)                        # 3.5, every point has the same eccentricity
cx.center(G)                        # every point: one Segment per edge
```

The triangle is a cycle of perimeter 7, so every point has eccentricity 7/2
and the center is the whole graph, returned as one `Segment` per edge. The
example also shows two facts about points of the same edge:

* their distance need not be the length of the segment between them, since the
  shortest route may leave the edge through one endpoint and return through the
  other (`2.2`, not `4.8`);
* the largest distance between two points of the edge `ab` is not `ℓ(ab)` but
  `(ℓ(ab) + d(a, b)) / 2`: here `3.5`, not `5`.

## Example

[`examples/emergency_network.ipynb`](examples/emergency_network.ipynb) runs the
code of Chapter 5 of the master's thesis on the weighted emergency network and
shows its output: distances, eccentricity, diameter, radius and center, and how
the continuous center differs from the discrete one.

## Building a continuous graph

A continuous graph is determined by the edge-weighted graph `(G, ℓ)` that
defines it, so no new combinatorial data is needed. The graph is built with
NetworkX as usual, with the length of each edge in the attribute `length`, and
wrapped in a `ContinuousGraph`. Another attribute can be chosen with
`ContinuousGraph(g, length="time")`, and an edge without it is given length 1.

* **Checks.** The graph must be simple, undirected and connected, and every
  length must be positive; otherwise an explicit error is raised.
* **A view, not a copy.** `ContinuousGraph` refers to `g` without copying it
  and provides no way of modifying it. It stores the distances between vertices
  once they are computed, so `g` should not be modified afterwards. A plain
  NetworkX graph can also be passed directly to any function of the package.
* **Points.** A point is a triple `(u, v, θ)`: the edge `uv` and the parameter
  `θ ∈ [0, 1]` measured from `u`. Points of a graph are built with
  `G.point(u, v, theta=...)` or `G.point(u, v, offset=...)`, the latter giving
  the distance `λ` from `u`, and `G.offset(p)` recovers `λ`. Every point is put
  in canonical form, so `Point("b", "a", 0.25) == Point("a", "b", 0.75)`, and
  points can be used as dictionary keys.
* **Exact arithmetic.** Lengths may be `float` or `fractions.Fraction`. The
  algorithms use only the four arithmetic operations and comparisons, so with
  `Fraction` lengths, and `ContinuousGraph(g, tol=0)`, every distance and
  parameter is an exact rational. Exact arithmetic is not imposed, since it is
  considerably slower.

## What it computes

| Function | Returns | Complexity |
| --- | --- | --- |
| `distance(G, p, q)` | `d(p, q)` | `O(1)` |
| `eccentricity(G, p)` | `ecc(p)` | `O(m)` |
| `eccentricity_function(G, u, v)` | `ecc` along `uv` | `O(m²)` |
| `diameter(G)`, `diametral_pair(G)` | `diam(𝒢)`, a diametral pair | `O(m²)` |
| `local_radius(G, u, v)`, `local_center(G, u, v)` | `rad(uv)`, local center of `uv` | `O(m²)` |
| `radius(G)`, `center(G)` | `rad(𝒢)`, `C(𝒢)` | `O(m³)` |
| `tree_diameter`, `tree_radius`, `tree_center` | parameters of a continuous tree | `O(n)` |
| `cycle_diameter`, `cycle_radius`, `cycle_center` | parameters of a continuous cycle | `O(n)` |

The costs assume that the distances between vertices are already computed,
which takes `O(nm log n)` time with Dijkstra's algorithm from every vertex;
they are computed once and stored. The functions for trees and cycles do not
need them. In this first version the upper envelope is stored in a Python
list, copied at each insertion, so an envelope costs `O(m²)` instead of the
`O(m log m)` of the 2–3 tree of Algorithm 1 (Hansen, Labbé and Nicolas), and
the radius and the center cost `O(m³)` instead of `O(m² log m)`.

The continuous center need not be finite: on a cycle it is the whole graph. It
is therefore returned as a list whose elements are either a `Point`, an
isolated central point, or a `Segment`, a stretch of an edge made of central
points.

## Structure of the package

```
continuumx/
  classes/      ContinuousGraph, Point, and Segment, used to return the center
  structures/   hat functions, piecewise linear functions and upper envelopes
  algorithms/   distance, eccentricity, diameter, radius, center, and families
                (closed forms for trees and cycles)
tests/          test suite
examples/       the emergency network notebook
```

The central part is `structures`. Every parameter is read off the hat
functions of each edge: the diameter is the largest of their ceilings, the
radius is the smallest of the minima of their upper envelopes, one for each
edge, and the center is the set of points where these envelopes attain the
radius.

## Development

Clone the repository and install it in editable mode, with the development
dependencies:

```bash
git clone https://github.com/AnaCarrascoM/continuumx
cd continuumx
pip install -e ".[dev]"
```

The test suite is run with `python -m pytest`, and it also runs automatically
on every change. It compares the results with:

1. the closed forms for trees and cycles;
2. the worked examples of Chapter 4 of the master's thesis, with `Fraction`
   lengths and zero tolerance;
3. a brute-force computation: every edge is subdivided into `k` equal pieces
   and NetworkX shortest paths are run on the result;
4. the outputs printed in Chapter 5 and stored in the example notebook.

It also checks properties that must hold on any graph, such as the metric
axioms, `rad ≤ diam ≤ 2·rad`, and invariance under subdivision of an edge
(Remark 4.2).

## Future work

As listed in Section 5.8 of the master's thesis:

* **Efficiency.** Store the upper envelope in an ordered structure, as the 2–3
  tree of Hansen, Labbé and Nicolas, to match the `O(m log m)` bound per edge.
* **Families of graphs.** Faster algorithms for graphs of bounded treewidth,
  planar graphs and cactus graphs.
* **Other parameters**, such as the mean distance or the dilation.
* **Shortest routes** between two points, not only their distance.
* **Visualization** of continuous graphs, with the computed points and
  segments marked on them.
* **Documentation**: a reference website generated with Sphinx.
* **Experimental tool**: testing conjectures on many graphs, such as whether
  the reduction with pendant paths used for the diameter also works for the
  radius (Open Problem 4.1).
* **Augmentation problems**: how the diameter or the radius changes when new
  edges are added, for instance which new road most reduces the worst-case
  response time in the emergency network.
* **Real networks**: road networks from OpenStreetMap through OSMnx, after
  some adaptation.

## Citation

If you use `continuumx`, please cite both the software and the master's
thesis it accompanies:

```bibtex
@misc{carrasco2026continuumx,
  author       = {Carrasco Martín, Ana},
  title        = {continuumx: a {Python} library for continuous graphs},
  year         = {2026},
  howpublished = {\url{https://github.com/AnaCarrascoM/continuumx}},
  note         = {Version 0.1.0}
}

@mastersthesis{carrasco2026distances,
  author = {Carrasco Martín, Ana},
  title  = {Computing Distances in Combinatorial and Continuous Graphs},
  school = {Universidad de Sevilla},
  year   = {2026},
  month  = nov,
  note   = {Supervised by Delia Garijo Royo}
}
```

## License

`continuumx` is released under the [MIT License](LICENSE).
