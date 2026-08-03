# .trx — Tractography Reference eXchange Format

`ritk-trx` reads and writes the TRX (Tractography Reference eXchange) format.
TRX stores tractography data as a directory containing a JSON header and raw
binary arrays, with explicit support for per-vertex data (DPV), per-streamline
data (DPS), and hierarchical grouping.

## Format layout

A TRX directory contains:

```text
mytracks.trx/
├── header.json      — JSON metadata
├── positions.raw    — flat array of point coordinates (nb_points × 3)
├── offsets.raw      — cumulative point indices (nb_streamlines + 1)
└── dpv/             — per-vertex data arrays (optional)
    ├── FA.raw
    └── MD.raw
```

### Positions and offsets

Points from all streamlines are flattened into a single `positions.raw` array
of `nb_points × 3` values. The `offsets.raw` array has `nb_streamlines + 1`
entries; streamline `i` occupies positions `offsets[i]..offsets[i+1]`. The
sentinel `offsets[nb_streamlines]` must equal `nb_points`.

This contiguous layout eliminates per-streamline count headers and enables
zero-copy slicing of the position array.

### Data types

The `dtype` field in `header.json` declares the binary encoding for positions.
Supported values are `"float32"` (4-byte LE f32) and `"float64"` (8-byte LE
f64). Offset arrays are always `uint64` LE. DPV/DPS arrays declare their own
`dtype` in the header's `dpv`/`dps` maps.

## Coordinate system

Streamlines are stored in physical millimetre coordinates — the native TRX
coordinate system. The optional `reference` field in the header carries NIfTI
reference metadata: an affine (16 f64 values), voxel dimensions, and voxel
sizes, enabling conversion to/from voxel space when a reference image is
available.

## Reading

`TrkTractogram::read_dir` reads from a directory path, expecting
`header.json`, `positions.raw`, and `offsets.raw`. If the header declares DPV
arrays, `dpv/<name>.raw` files are also read:

```rust,ignore
use ritk_trx::TrxTractogram;

let trx = TrxTractogram::read_dir("mytracks.trx")?;
println!("{} streamlines, {} total points",
    trx.streamlines.len(),
    trx.header.nb_points,
);
```

`from_raw` and `from_raw_with_dpv` read from in-memory buffers, enabling
round-trip fidelity through encode/decode cycles.

### Validation

The reader checks:
- `positions.raw` length matches `nb_points × 3` for the declared dtype
- `offsets.raw` has exactly `nb_streamlines + 1` entries
- The sentinel offset equals `nb_points`
- Every offset is monotonic and within bounds
- Every coordinate is finite
- Every polyline is valid Gaia geometry

## Writing

`TrkTractogram::write_dir` creates the directory structure, writes
`header.json`, `positions.raw`, `offsets.raw`, and any DPV data files:

```rust,ignore
trx.write_dir("output.trx")?;
```

`to_raw` returns the header and raw byte buffers for in-memory encoding:

```rust,ignore
let (header, positions, offsets, dpv_data) = trx.to_raw()?;
```

## Header fields

`TrxHeader` is a Serde-serialisable JSON structure:

| Field | Type | JSON key | Meaning |
|---|---|---|---|
| `nb_streamlines` | `u64` | `nb_streamlines` | Total streamline count |
| `nb_points` | `u64` | `nb_points` | Total point count |
| `dimensions` | `u32` | `dimensions` | Spatial dimensions (always 3) |
| `dtype` | `String` | `dtype` | Positions encoding ("float32" or "float64") |
| `reference` | `Option<TrxReference>` | `reference` | NIfTI reference metadata |
| `dpv` | `HashMap<String, TrxArrayDef>` | `dpv` | Per-vertex data arrays |
| `dps` | `HashMap<String, TrxArrayDef>` | `dps` | Per-streamline data arrays |
| `dpg` | `HashMap<String, TrxArrayDef>` | `dpg` | Per-group data |
| `groups` | `Vec<TrxGroup>` | `groups` | Named streamline groups |

### DPV and DPS arrays

`TrxArrayDef` declares one data array:

| Field | Type | Meaning |
|---|---|---|
| `dtype` | `String` | Data type (e.g. `"float32"`, `"int32"`) |
| `n_components` | `u32` | Components per element (1 = scalar, 3 = vector) |

DPV data maps key names to raw byte buffers in `TrxTractogram::dpv_data`. Each
buffer must match the length implied by `nb_points × n_components ×
sizeof(dtype)`.

### NIfTI reference

`TrxReference` carries optional NIfTI image metadata:

| Field | Type | Meaning |
|---|---|---|
| `path` | `Option<String>` | Path to the reference file |
| `affine` | `Option<[f64; 16]>` | 4×4 row-major voxel→physical affine |
| `dimensions` | `Option<[u32; 3]>` | Voxel grid dimensions |
| `voxel_sizes` | `Option<[f64; 3]>` | Voxel size in mm |

### Groups

`TrxGroup` defines a named collection of streamline indices, enabling
hierarchical organisation of tractography output (e.g. by bundle):

```json
{
  "name": "CST_left",
  "indices": [0, 1, 2, 3, 4]
}
```

## Error types

`TrxError` covers every failure mode:

| Variant | Condition |
|---|---|
| `Io` | File I/O error |
| `Json` | JSON (de)serialisation failure |
| `PositionsLengthMismatch` | `positions.raw` length ≠ `nb_points × 3` |
| `OffsetsLengthMismatch` | `offsets.raw` length ≠ `nb_streamlines + 1` |
| `SentinelMismatch` | Last offset ≠ `nb_points` |
| `InvalidOffset` | Non-monotonic or out-of-bounds offset |
| `UnsupportedDtype` | Unrecognised dtype string |
| `NonFiniteCoordinate` | NaN or infinite coordinate |
| `InvalidPolyline` | Gaia rejected the point sequence |

## Usage in the diffusion pipeline

`ritk-tractography` provides `TractographyResult::to_trx` and
`to_trx_with_dpv` that produce `TrxTractogram` values with per-vertex data
arrays (e.g. FA, MD). The caller populates the header's `dpv` map with
`TrxArrayDef` entries and provides the corresponding raw byte buffers. See
the [Tractography](tractography.md) chapter.

## Comparison with .trk and .tck

| Property | `.trk` | `.tck` | `.trx` |
|---|---|---|---|
| Storage | Single binary file | Single binary file | Directory of files |
| Header | Fixed 1000-byte binary | Text key:value pairs | JSON |
| Coordinates | Voxel (affine→RAS) | Scanner mm | Physical mm |
| Per-point scalars | Inline after positions | Weights sidecar only | DPV arrays |
| Per-streamline props | Inline after streamline | Not supported | DPS arrays |
| Groups | Not supported | Not supported | Named index groups |
| Reference image | Implicit (affine) | Transform header key | Explicit NIfTI reference |
