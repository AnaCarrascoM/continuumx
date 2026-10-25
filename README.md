# continuumx

`continuumx` is a Python library for **continuous graphs**, built on top of
[networkx](https://networkx.org/). Where a classical graph only has distances
between vertices, a continuous graph reads every edge as a segment of
prescribed length, so distances are defined between *any* two points of the
network — not just its vertices.

It implements the constructions and algorithms of the master's thesis *Metric
Parameters of Continuous Graphs* (A. Carrasco Martín, Universidad de Sevilla).
It is a **prototype**: version 0.1 covers distance, eccentricity, diameter,
radius and centre, exactly and for arbitrary connected weighted graphs.

## Installation

```bash
pip install -e ".[dev,drawing]"
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

cx.distance(G, p, q)                # 2.2  — not 4.8: the path leaves the edge
cx.diameter(G)                      # 3.5  — not 5.0: half the perimeter
cx.radius(G)                        # 3.5
cx.center(G)                        # the whole graph, as a list of Segments
```

The package namespace already says "continuous", so the parameters keep their
`networkx` names: comparing the two settings is `nx.diameter(g)` against
`cx.diameter(g)`.

## What it computes

| Function | Returns | Complexity |
| --- | --- | --- |
| `distance(G, p, q)` | `d(p, q)` between two points | `O(1)` after the shortest paths |
| `eccentricity_to_edge(G, p, u, v)` | `max` distance from `p` to an edge | `O(1)` |
| `eccentricity(G, p)` | `ecc(p)` | `O(m)` |
| `eccentricity_function(G, u, v)` | `θ ↦ ecc(x_θ)` as a piecewise-linear function | `O(m log m)` |
| `diameter(G)`, `diametral_pair(G)` | the diameter and one witness pair | `O(nm log n + m²)` |
| `local_radius(G, u, v)`, `local_center(G, u, v)` | the LCR and LCC of one edge | `O(m log m)` |
| `radius(G)`, `center(G)` | the radius and the centre | `O(nm log n + m² log m)` |
| `tree_*`, `cycle_*` | closed forms for trees and cycles | linear |

The centre and the sets of central points are returned as lists of
`Segment` objects: unlike the discrete centre, the continuous centre is in
general infinite (on a cycle it is the whole graph) and need not contain a
vertex.

## Design notes

**A view, not a subclass.** A continuous graph adds no combinatorial data to a
graph — it is a pair `(G, ℓ)` read differently — so `ContinuousGraph` wraps an
`nx.Graph` instead of subclassing it. Subclassing would inherit mutation, which
cannot preserve the invariants (connectivity, positive lengths); wrapping gives
a home for the shortest-path cache every continuous algorithm needs. Every
function also accepts a plain `nx.Graph` and wraps it internally.

**Points carry `θ`, not `λ`.** A point is a triple `(u, v, θ)` with
`θ ∈ [0, 1]` measured from `u`. The normalised parameter is *intrinsic* — a
triple can be validated without knowing the edge length — and it is the
parametrisation the theory uses. The absolute offset is `G.offset(p)`. Points
are normalised on construction, so `Point("b", "a", 0.25) == Point("a", "b",
0.75)` and `Point("a", "b", 0.0) == Point("a", "c", 0.0)`, and points are
hashable.

**One computational core.** Every parameter is a reading of the same object:
the eccentricity along an edge is the upper envelope of `m + 1` *hat
functions* (`continuumx.structures.piecewise`). The diameter is the largest
ceiling; the radius is the smallest minimum; the centre is where those minima
are attained. Adding a parameter means reading an envelope differently, not
writing new formulas.

**Exact arithmetic.** The core uses only `+`, `-`, `min`, `max` and division by
2 — no `math`, no `numpy`. With `fractions.Fraction` edge lengths every
computed parameter is an exact rational and `tol=0` is safe, which matters for
the centre, whose structure is decided by equality tests.

**Deliberate restrictions in 0.1.** Multigraphs, self-loops and zero-length
edges are rejected with an explicit error rather than mishandled silently.
Parallel edges are a genuine loss (two edges of different lengths between the
same pair form a continuous `C₂`) but the triple `(u, v, θ)` cannot name one of
them; supporting them later means adding an edge key to `Point`, not rewriting
the algorithms.

## Two traps this library avoids

Both come up in any first implementation and both give plausible wrong numbers.
The triangle with `ℓ(ab) = 5` and a path `a c b` of total length `2` exhibits
them:

* the distance between two points of the **same** edge is *not*
  `|λ_p − λ_q|` — a path may leave through one endpoint and return through the
  other (`2.2`, not `4.8`);
* the largest distance between two points of the same edge is *not* `ℓ(e)` —
  here it is `3.5`, not `5`.

## Project layout

```
continuumx/
  classes/      ContinuousGraph (an immutable view over nx.Graph), Point, Segment
  structures/   PiecewiseLinear, HatFunction, upper_envelope — the computational core
  algorithms/   distance, eccentricity, diameter, radius, center, families
  drawing/      matplotlib rendering of points and segments
tests/          pytest suite, mirrors the package layout
```

## Testing

```bash
python -m pytest
```

The suite validates the library three ways:

1. **An independent oracle.** Every edge is subdivided into `k` equal pieces and
   plain `networkx` shortest paths are run on the result. Distances between
   sampled points are then exact, and the sampled diameter and radius bracket
   the true ones within `max_edge_length / k`.
2. **Closed forms.** Trees (`diam(𝒯) = diam(T)`, `rad = diam/2`, centre a single
   point) and cycles (`diam = rad = Per/2`, centre the whole graph).
3. **Structural properties** on random graphs: the metric axioms,
   `rad ≤ diam ≤ 2·rad`, linear scaling under a change of unit, and
   **invariance under subdivision** — splitting an edge in two yields an
   isometric continuous graph, so no parameter may change. That last one
   exercises the point representation, the distances, the hat functions and the
   envelopes at once.

## Not implemented

* The `O(n log^{4k−2} n)` diameter algorithm for graphs of bounded treewidth,
  and the `O(nF log n)` algorithm for planar graphs. Both need heavy machinery
  (orthogonal range searching, face structure) whose constants make them slower
  than the `O(m²)` algorithm at any size a prototype will meet.
* Multigraphs, self-loops, zero-length edges (see above).
* Parameters whose continuous theory is not settled.
