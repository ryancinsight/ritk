# ritk-connectome

Connectome construction and graph analysis: turning a tractogram plus a
parcellation into the graph of which brain regions connect, and measuring it.

```text
Parcellation (labels) ──┐
                        ├──► ConnectivityMatrix ──► graph measures
Streamlines (polylines) ┘
```

## What it provides

**Construction** decides two things, and both change the answer.

- *Which region does an endpoint belong to?* Reading the label directly under
  the endpoint discards most of a tractogram, because tracking stops at the
  grey/white boundary while a cortical parcellation labels only grey matter.
  `EndpointAssignment::RadialSearch` assigns to the nearest labelled voxel
  within a radius, recovering those without moving an endpoint already inside a
  region.
- *What does an edge weigh?* `EdgeWeighting` offers streamline count, inverse
  pathway length, inverse node volume, and mean length. The two normalisations
  divide out known geometric confounds. None of them makes a count a
  measurement of connection strength.

Every streamline lands in one of five `StreamlineAccounting` buckets that sum to
the whole, because a matrix built from a tractogram four fifths discarded is a
different claim from one built from a twentieth, and the weights do not say
which you have.

**Measures** cover clustering (binary and Onnela-weighted), shortest paths,
characteristic path length with its reachable fraction, global and local
efficiency, Brandes betweenness centrality, Louvain communities with weighted
modularity, rich-club coefficients, and connected components.

Two things worth knowing about them:

- Weights are *strengths* and shortest paths need *distances*, so the reciprocal
  is taken at one point in the crate. Getting that backwards inverts every
  derived measure while leaving plausible numbers.
- Community detection visits nodes in index order rather than at random. A
  measure that changes when recomputed cannot be compared between subjects.

## Example

```rust
use gaia::Polyline;
use leto::geometry::Point3;
use ritk_connectome::{ConnectomeConfig, build_connectivity_matrix};
use ritk_parcellation::{Parcellation, ParcellationGrid};

let grid = ParcellationGrid::axis_aligned([4, 1, 1], [1.0; 3], [0.0; 3])?;
let parcellation = Parcellation::new(Box::new([1, 0, 0, 2]), grid, Vec::new())?;

let streamline = Polyline::new(vec![
    Point3::new(0.0, 0.0, 0.0),
    Point3::new(3.0, 0.0, 0.0),
])?;

let matrix = build_connectivity_matrix(
    &parcellation,
    std::slice::from_ref(&streamline),
    &ConnectomeConfig::default(),
)?;

assert_eq!(matrix.weight(1, 2), Some(1.0));
let measures = matrix.measures();
assert_eq!(measures.edge_count(), 1);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Interpretation

An edge weight is a property of the tractogram, not of the brain. Tracking
systematically favours short, straight, high-anisotropy paths and loses long,
curved, or crossing ones. Comparison across subjects processed identically is
the defensible use; absolute interpretation of a single weight is not.

## Related crates

- [`ritk-parcellation`](../ritk-parcellation) owns the label volume.
- [`ritk-tractography`](../ritk-tractography) produces the streamlines.
- [`ritk-diffusion`](../ritk-diffusion) fits the orientation models they follow.

## Documentation

The RITK book: <https://ryancinsight.github.io/ritk/>.

## Licence

MIT OR Apache-2.0.
