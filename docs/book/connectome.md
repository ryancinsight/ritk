# Connectome Construction and Graph Measures

`ritk-connectome` is the third leaf crate from
[ADR 0036 decision 1](https://github.com/ryancinsight/atlas/blob/main/docs/adr/0036-neuroimaging-and-mr-ownership.md).
It reduces a set of streamline endpoints and a volumetric parcellation into a
weighted undirected `ConnectivityMatrix` — the adjacency of anatomical regions
— plus per-node and global graph measures.

## Data flow

```text
Parcellation (label volume) ──┐
                              ├──► ConnectivityMatrix ──► graph measures
Streamlines (Gaia polylines) ─┘
```

Each streamline contributes one unit of weight to the edge connecting the two
parcellation regions its endpoints fall in. Streamlines whose endpoints land
in the same region, or outside the parcellation volume, are counted but do
not add an inter-region edge.

The crate sits at the end of the diffusion-MRI pipeline documented in this
part: the [Gradient Schemes](diffusion_scheme.md) chapter validates
acquisition metadata, the [Diffusion Models](ritk_diffusion.md) chapter fits
orientation models, the [Tractography](tractography.md) chapter produces
streamlines, and this chapter builds the connectome graph from those
streamlines.

## Parcellation

A `Parcellation` is a 3-D label volume where each voxel carries one region
ID. The special label `0` conventionally represents background or outside
the brain. Labels are stored in z-major (slice-first) order.

### Construction

`Parcellation::new` validates that the label array length matches the
declared shape, that every dimension is nonzero, and that at least one
non-background label exists:

```rust,ignore
use ritk_connectome::Parcellation;

let parcellation = Parcellation::new(
    labels,                      // Box<[u32]>
    [256, 256, 128],             // shape [nx, ny, nz]
    [1.0, 1.0, 1.0],             // spacing [sx, sy, sz] in mm
    [0.0, 0.0, 0.0],             // origin [ox, oy, oz] in mm
    vec![(1, "Cortex".into())],   // region_names
)?;
```

The `region_names` field associates human-readable names with label IDs.
The background label `0` may appear here (e.g. `"Background"`) or be
omitted.

### Spatial query

`label_at(point)` maps a physical point to its region label via
nearest-neighbour voxel lookup. It returns `None` when the point is
outside the grid or any coordinate is non-finite:

```rust,ignore
let label = parcellation.label_at(&point);
```

### Accessors

| Method | Returns |
|---|---|
| `shape()` | `[nx, ny, nz]` |
| `spacing()` | `[sx, sy, sz]` in mm |
| `origin()` | `[ox, oy, oz]` in mm |
| `region_labels()` | Sorted list of non-background labels |
| `region_count()` | Number of unique non-background regions |
| `region_names()` | `&[(u32, String)]` — label-to-name mapping |

## Connectivity Matrix

`ConnectivityMatrix` is a weighted undirected adjacency between
parcellation regions. Weights are streamline counts — each streamline
contributes 1.0 to exactly one edge. Weights are stored as a flat
\\(n \\times n\\) row-major matrix with an upper-triangular convention:
edge \\((i, j)\\) with \\(i \\le j\\) is at index \\(i \\cdot n + j\\).

### Construction

`build_connectivity_matrix` takes a `Parcellation` and a slice of Gaia
`Polyline<f64>` values and produces a `ConnectivityMatrix`:

```rust,ignore
use ritk_connectome::build_connectivity_matrix;

let matrix = build_connectivity_matrix(&parcellation, &streamlines)?;
```

For each streamline the first and last points are mapped to parcellation
region labels. The algorithm handles three cases:

| Case | Action |
|---|---|
| Both endpoints in different non-background regions | Increment edge \\((a, b)\\) |
| Both endpoints in the same non-background region | Increment `intra_region_count` and the self-edge |
| One or both endpoints background or out of bounds | Increment `skipped_count` |

### Diagnostics

| Method | Meaning |
|---|---|
| `region_count()` | Number of regions |
| `region_labels()` | Region labels in internal index order |
| `total_streamlines()` | Streamlines that contributed (≥ 2 points) |
| `intra_region_count()` | Streamlines whose endpoints landed in the same region |
| `skipped_count()` | Streamlines with out-of-bounds or background endpoints |

### Edge query

`weight(source, target)` returns the streamline count for an edge. Order
is irrelevant (undirected graph). Returns `None` when either label is not
in the matrix:

```rust,ignore
let w = matrix.weight(1, 2); // Some(count) or None
```

`edges()` iterates over all edges with nonzero weight, returning
`ConnectivityEdge` structs with `source`, `target`, and `weight` fields.
Self-edges are included; filter by `edge.source != edge.target` to
exclude them.

`edge_count()` returns the number of edges with nonzero weight, excluding
self-edges.

## Graph measures

### Degree and strength

`degree(label)` returns the binary degree — the number of distinct
neighbour regions connected by at least one streamline. Returns `None`
when the label is unknown:

```rust,ignore
let d = matrix.degree(1); // Some(count) or None
```

`strength(label)` returns the weighted degree — the sum of all streamline
counts on incident edges (excluding the self-edge, which would be counted
twice in the upper-triangular representation):

```rust,ignore
let s = matrix.strength(1); // Some(sum) or None
```

### Density

`density()` returns the graph density — the ratio of actual edges to
possible edges. For an undirected graph with \\(n\\) nodes, the maximum is
\\(n(n-1)/2\\). Density ∈ \\([0, 1]\\) for \\(n > 1\\); returns `0.0` when
\\(n \\le 1\\):

```rust,ignore
let rho = matrix.density(); // ∈ [0, 1]
```

## FreeSurfer surface annotation reader

The `freesurfer` module reads FreeSurfer surface-based parcellation
formats. A `SurfaceAnnotation` must be rasterised onto a volumetric
`Parcellation` before it can feed the connectivity matrix pipeline — that
conversion requires surface geometry (vertex coordinates and faces) and is
not included in this first increment.

### .annot binary format

`SurfaceAnnotation::read` parses the FreeSurfer `.annot` binary format:

- Magic number `-2` (i32 LE)
- Vertex count (i32 LE)
- Label table: structure index, name length, null-terminated name, RGBA
- Per-vertex label indices (i32 LE × vertex count)
- Additional colour table (same layout) and per-vertex colour values

```rust,ignore
use ritk_connectome::freesurfer::SurfaceAnnotation;

let annotation = SurfaceAnnotation::read(file)?;
println!("{} vertices, {} label entries",
    annotation.vertex_count,
    annotation.label_table.len(),
);
```

The structure index carries the canonical FreeSurfer label ID. When the
index is zero (conventionally the first `"Unknown"` entry), the table
index is used as a fallback.

### Colour lookup table

`read_freesurfer_lut` reads the plain-text `FreeSurferColorLUT.txt`
format:

```text
# FreeSurfer Color Lookup Table
0   Unknown   0   0   0   0
1   Cortical-Gray-Matter   205  62  78  0
2   Cortical-White-Matter  0   225 0   0
```

Returns `Vec<(u32, String)>` suitable for `Parcellation::new` as
`region_names`:

```rust,ignore
use ritk_connectome::freesurfer::read_freesurfer_lut;

let names = read_freesurfer_lut(file)?;
```

Lines starting with `#` are comments and blank lines are skipped.

## JSON serialisation

`ConnectivityMatrix` implements Serde `Serialize` and `Deserialize`.
`to_json()` and `from_json()` provide a lightweight codec for
interchange:

```rust,ignore
let json = matrix.to_json()?;
let restored = ConnectivityMatrix::from_json(&json)?;
```

Connectivity matrices persist through Consus formats per ADR 0036
decision 2; Consus HDF5 integration is the natural next step.

## Error types

`ConnectomeError` covers every failure mode in the crate:

| Variant | Condition |
|---|---|
| `EmptyParcellation(u32)` | Parcellation has only background voxels |
| `UnknownRegion { label }` | A region label referenced by an edge does not exist |
| `RegionCountMismatch { expected, actual }` | Label array length ≠ shape product, or zero-dimension grid |
| `EndpointOutOfBounds { streamline_index, endpoint, x, y, z }` | A streamline endpoint falls outside the parcellation volume |
| `Json(serde_json::Error)` | JSON serialisation or deserialisation failed |

`FreeSurferSurfaceError` (in the `freesurfer` module) covers I/O errors,
invalid magic bytes, malformed label table entries, and unreasonable
vertex counts.

## What the current increment establishes

The crate establishes deterministic graph construction from a label volume
and streamline set, upper-triangular weighted storage, per-node degree and
strength, and global graph density. It does not establish the biological
validity of the resulting graph — edge weights are raw streamline counts
rather than connection probabilities, and no length correction,
thresholding, or null model is applied. Those concerns are downstream of
this crate's ADR 0036 remit.
