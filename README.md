# continuumx

`continuumx` is a Python library for **continuous graphs**, built on top of
[NetworkX](https://networkx.org/). Where a classical graph only has distances
between vertices, a continuous graph reads every edge as a segment of
prescribed length, so distances are defined between *any* two points of the
graph, not just its vertices.

It implements the constructions and algorithms of the master's thesis
*Computing Distances in Combinatorial and Continuous Graphs* (A. Carrasco
Martín, Universidad de Sevilla, 2026). It is a **prototype**: version 0.1
covers distance, eccentricity, diameter, radius and center, for arbitrary
connected graphs with positive edge lengths.

## Installation

`continuumx` requires Python 3.9 or later and NetworkX 3.0 or later. Install it
directly from the repository:

```bash
pip install git+https://github.com/AnaCarrascoM/continuumx
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
cx.diameter(G)                      # 3.5
cx.radius(G)                        # 3.5
cx.center(G)                        # the whole graph, as a list of Segments
```

This triangle shows two facts about points of the same edge that are easy to
overlook:

* the distance between two points of the same edge is not the length of the
  segment between them: a path may leave through one endpoint and return
  through the other (`2.2`, not `4.8`);
* the largest distance between two points of the same edge is not `ℓ(ab)` but
  `(ℓ(ab) + d(a, b)) / 2`: here `3.5`, not `5`.

The package namespace already says "continuous", so the parameters keep their
NetworkX names: comparing the two settings is `nx.diameter(g)` against
`cx.diameter(g)`.

## Examples

[`examples/emergency_network.ipynb`](examples/emergency_network.ipynb) walks
through the package on the weighted emergency network, following Chapter 5 of
the thesis: distances, eccentricity, diameter, radius and center, and how the
continuous center differs from the discrete one.

## What it computes

| Function | Returns | Complexity |
| --- | --- | --- |
| `distance(G, p, q)` | `d(p, q)` between two points | `O(1)` |
| `eccentricity_to_edge(G, p, u, v)` | the eccentricity of `p` relative to the edge `uv` | `O(1)` |
| `eccentricity(G, p)` | `ecc(p)` | `O(m)` |
| `eccentricity_function(G, u, v)` | `θ ↦ ecc(x_θ)` as a piecewise-linear function | `O(m²)` |
| `diameter(G)`, `diametral_pair(G)` | the diameter and one diametral pair | `O(nm log n + m²)` |
| `local_radius(G, u, v)`, `local_center(G, u, v)` | the local radius and local center of one edge | `O(m²)` |
| `radius(G)`, `center(G)` | the radius and the center | `O(nm log n + m³)` |
| `tree_*`, `cycle_*` | closed forms for trees and cycles | `O(n)` |

`O(nm log n)` is the cost of the shortest paths between vertices (Dijkstra from
every vertex), computed once and cached; the other costs assume they are
available. The upper envelope is stored in a Python list, which is copied at
each insertion, so an envelope costs `O(m²)` rather than the `O(m log m)` that
an ordered structure (a 2–3 tree, as in Hansen, Labbé and Nicolas) would give.

The center is returned as a list whose elements are either a `Point`, an
isolated central point, or a `Segment`, a stretch of an edge made of central
points: unlike the discrete center, the continuous center is in general
infinite (on a cycle it is the whole graph) and need not contain a vertex.

## Design notes

**A view, not a subclass.** A continuous graph adds no combinatorial data to a
graph: it is a pair `(G, ℓ)` read differently. So `ContinuousGraph` wraps an
`nx.Graph` instead of subclassing it. Subclassing would inherit mutation, which
cannot preserve the invariants (connectivity, positive lengths); wrapping gives
a home for the shortest-path cache every continuous algorithm needs. Every
function also accepts a plain `nx.Graph` and wraps it internally.

**Points carry `θ`, not `λ`.** A point is a triple `(u, v, θ)` with
`θ ∈ [0, 1]` measured from `u`. The normalized parameter is *intrinsic*, since
a triple can be validated without knowing the edge length, and it is the
parametrization the theory uses. The absolute offset is `G.offset(p)`. Points
are normalized on construction, so `Point("b", "a", 0.25) == Point("a", "b",
0.75)` and `Point("a", "b", 0.0) == Point("a", "c", 0.0)`, and points are
hashable.

**One computational core.** Every parameter is a reading of the same object:
the eccentricity along an edge is the upper envelope of `m + 1` *hat
functions* (`continuumx.structures.piecewise`). The diameter is the largest
ceiling; the radius is the smallest minimum; the center is where those minima
are attained.

**Exact arithmetic.** The core uses only the four arithmetic operations, `min`
and `max`, with no square roots or other functions. With `fractions.Fraction`
edge lengths every computed parameter is an exact rational and `tol=0` is safe,
which matters for the center, whose structure is decided by equality tests.

**Restrictions in 0.1.** As in the thesis, graphs are simple, undirected and
connected, with positive edge lengths; multigraphs, self-loops and zero-length
edges are rejected with an explicit error. Two parallel edges of different
lengths would define a continuous cycle `C₂`, but the triple `(u, v, θ)` cannot
name one of them; supporting them would mean adding an edge key to `Point`.

## Development

Clone the repository and install it in editable mode, with the development
dependencies:

```bash
git clone https://github.com/AnaCarrascoM/continuumx
cd continuumx
pip install -e ".[dev]"
```

### Project layout

```
continuumx/
  classes/      ContinuousGraph (an immutable view over nx.Graph), Point, Segment
  structures/   PiecewiseLinear, HatFunction, upper_envelope: the computational core
  algorithms/   distance, eccentricity, diameter, radius, center, families
tests/          pytest suite
examples/       notebooks walking through the package
```

### Testing

```bash
python -m pytest
```

The suite, which also runs the examples in the docstrings, validates the
library in four ways:

1. **An independent oracle.** Every edge is subdivided into `k` equal pieces and
   plain NetworkX shortest paths are run on the result. Distances between
   sampled points are then exact, and the sampled diameter and radius bracket
   the true ones within `max_edge_length / k`.
2. **Closed forms.** Trees (`diam(𝒯) = diam(T)`, `rad = diam/2`, center a single
   point) and cycles (`diam = rad = Per/2`, center the whole graph).
3. **Structural properties** on random graphs: the metric axioms,
   `rad ≤ diam ≤ 2·rad`, linear scaling under a change of unit, and
   **invariance under subdivision**: splitting an edge in two yields an
   isometric continuous graph, so no parameter may change.
4. **The thesis.** The worked examples of Chapter 4 are checked with
   `Fraction` lengths and zero tolerance, and the outputs printed in Chapter 5
   are checked literally (`tests/test_chapter5.py`).

## Future work

As listed in Section 5.8 of the thesis:

* **Efficiency.** Store the upper envelope in an ordered structure, as the 2–3
  tree of Hansen, Labbé and Nicolas, to match the `O(m log m)` bound per edge.
* **Families of graphs.** Faster algorithms for graphs of bounded treewidth,
  planar graphs and cactus graphs.
* **Other parameters**, such as the mean distance or the dilation.
* **Shortest routes** between two points, not only their distance.
* **Visualization** of continuous graphs, with the computed points and
  segments marked on them.
* **Documentation**: a reference website generated with Sphinx.
* **Real networks**: road networks from OpenStreetMap through OSMnx.

## Citation

If you use `continuumx`, please cite the master's thesis it accompanies:

```bibtex
@mastersthesis{carrasco2026continuumx,
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
