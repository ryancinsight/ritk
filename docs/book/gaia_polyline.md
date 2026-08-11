# Gaia Polyline Geometry

Gaia owns the `Polyline` type — the canonical open curve that
`ritk-tractography` produces as streamline output and `ritk-connectome`
consumes as the input streamlines for connectivity matrix construction.
This is an upstream provider contract per [ADR 0036 decision
2](https://github.com/ryancinsight/atlas/blob/main/docs/adr/0036-neuroimaging-and-mr-ownership.md):
streamlines are
Gaia polyline geometry; RITK owns the integration policy that produces them
and does not define a local curve or polyline type.

## Why Gaia?

Gaia is the Atlas geometry owner — it already carries the `Point3<T>` type
(via Leto), exact geometric predicates, axis-aligned bounding boxes, and
mesh builders for CFD milifluidic simulation. Polyline geometry sits
naturally in the same domain: ordered point sequences in three dimensions,
with validation, measurement, and spatial queries. Placing `Polyline` here
means:

- Tractography output is typed in a geometry crate that downstream consumers
  (visualisation, meshing, surface reconstruction) already depend on.
- The same `Aabb<T>` spatial query is available for streamlines and meshes.
- Format-crate readers (`ritk-trk`, `ritk-tck`, `ritk-trx`) return
  `Vec<Polyline<f64>>` — the interchange format boundary is clean.

A RITK-local line or curve type is a boundary violation under ADR 0036.

## Polyline Construction

`Polyline::new(points)` accepts a `Vec<Point3<T>>` and validates:

- **At least two points.** A single-point polyline is geometrically a
  point, not a curve; a zero-point polyline is meaningless.
- **Every coordinate is finite.** NaN or infinity in any vertex coordinate
  is rejected with the offending index.

```rust,ignore
use gaia::Polyline;
use leto::geometry::Point3;

let line = Polyline::<f64>::new(vec![
    Point3::new(0.0, 0.0, 0.0),
    Point3::new(3.0, 4.0, 0.0),
])?;
assert_eq!(line.segment_count(), 1);
assert!((line.arc_length() - 5.0).abs() < f64::EPSILON);
```

The type parameter `T: Scalar` defaults to `f64`. Storage is frozen as a
boxed slice — no retained overcapacity — so memory never exceeds the vertex
count once validated.

### Error Types

`PolylineError` has two non-exhaustive variants:

| Variant | Condition |
|---|---|
| `TooFewPoints(usize)` | Fewer than two points provided |
| `NonFinitePoint { index }` | The first vertex with NaN/infinity |

## Accessors

| Method | Returns | Description |
|---|---|---|
| `len() -> usize` | Vertex count | At least 2 after successful construction |
| `is_empty() -> bool` | Always `false` | Maintains the collection-style contract |
| `segment_count() -> usize` | `len() - 1` | Number of straight connecting segments |
| `points() -> &[Point3<T>]` | All vertices | Immutable borrow, ordered |
| `first() -> Point3<T>` | First vertex | Panic-free; invariant ensures at least 2 points |
| `last() -> Point3<T>` | Last vertex | Panic-free; invariant ensures at least 2 points |
| `segments() -> impl Iterator` | Consecutive endpoint pairs | `windows(2)` over `points()` |

```rust,ignore
let streamline: Polyline<f64> = /* from tractography */;
for (from, to) in streamline.segments() {
    let length = (to - from).norm();
    // ... per-segment logic
}
```

## Measurements

### Arc Length

`arc_length()` computes the sum of Euclidean distances between consecutive
vertices — the physical length of the streamline in millimetres:

```rust,ignore
let length: f64 = streamline.arc_length();
```

This is the quantity used by tractography termination (minimum length
check) and connectome edge-weighting.

### Axis-Aligned Bounding Box

`aabb()` returns the `Aabb<T>` enclosing all vertices:

```rust,ignore
let bounds = streamline.aabb();
// bounds.min / bounds.max — Point3<T>
// bounds.extents() — (width, height, depth)
// bounds.center() — geometric centre
// bounds.volume() — enclosed box volume
```

The Gaia `Aabb<T>` type also supports point containment, AABB intersection,
and union operations. Tractography uses it for spatial indexing;
connectome verification uses it to check streamlines against the
parcellation volume.

## Streamline Output in ritk-tractography

`ritk-tractography` defines a `Streamline` struct that wraps a
`Polyline<f64>` plus per-point scalars:

```rust,ignore
pub struct Streamline {
    pub geometry: Polyline<f64>,
    pub scalars: Option<Box<[f64]>>,
}
```

The integration step converts accumulated integration points into a
`Polyline` through `points_to_polyline()`, which validates the point
sequence and returns a `TractographyError` on construction failure.

Export methods map `Streamline` to Gaia polylines for format-crate writers:

```rust,ignore
// .trk export (via ritk_trk)
let streamlines: Vec<Polyline<f64>> = self.streamlines
    .iter().map(|s| s.geometry().clone()).collect();
let trk = TrkTractogram::from_polylines(streamlines, header)?;

// .tck export (via ritk_tck) — same pattern
// .trx export (via ritk_trx) — same pattern
```

Each export method rips the geometry out of the `Streamline` wrapper and
hands a `Vec<Polyline<f64>>` to the format-crate writer, which stores
vertex coordinates in the target format's coordinate system and binary
encoding.

## Connectivity Matrix Input in ritk-connectome

`ritk-connectome::build_connectivity_matrix` takes `&[Polyline<f64>]` as
its streamline input:

```rust,ignore
use ritk_connectome::build_connectivity_matrix;

let matrix = build_connectivity_matrix(&parcellation, &streamlines)?;
```

The function samples each streamline's vertices against a volumetric
parcellation, mapping each transition between parcellation regions to an
edge in the connectivity matrix. The `first()` and `last()` accessors
determine endpoint regions for ROI-to-ROI connectivity; all intermediate
vertices determine the path.

## Relationship to the Aabb Type

Polyline and `Aabb` compose naturally for spatial queries:

```rust,ignore
// Quick reject: does the streamline intersect the parcellation volume?
if !parcellation_bbox.intersects(&streamline.aabb()) {
    continue; // skip this streamline entirely
}

// Per-vertex containment test
for point in streamline.points() {
    if parcellation_bbox.contains_point(&point) {
        let label = parcellation.label_at_point(&point);
        // ...
    }
}
```

This spatial pipeline — AABB cull → per-vertex query — is the core of the
connectome construction algorithm.

## Boundary

Under ADR 0036 decision 2, the polyline type and its geometric predicates
belong to Gaia, never to RITK. RITK owns the integration and termination
policy that produces `Polyline<f64>` instances from a direction field; it
does not define a local polyline, curve, or line type. The format-crate
readers and the connectome builder consume `Polyline<f64>` as their
canonical curve representation.

## References

- [ADR 0036 — Neuroimaging and MR Ownership](https://github.com/ryancinsight/atlas/blob/main/docs/adr/0036-neuroimaging-and-mr-ownership.md)
- [Deterministic Streamline Tractography](tractography.md) — the RITK integration side
- [Connectome Construction and Graph Measures](connectome.md) — the RITK consumption side
- [.trk Format](trk_format.md), [.tck Format](tck_format.md), [.trx Format](trx_format.md) — interchange format chapters
