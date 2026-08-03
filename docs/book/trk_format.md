# .trk — DSI Studio / TrackVis Tractogram Format

`ritk-trk` reads and writes the DSI Studio / TrackVis `.trk` binary tractogram
format. The format stores a fixed 1000-byte header followed by per-streamline
data blocks, with an embedded voxel-to-RAS affine that maps on-disk voxel
coordinates to physical space.

## Format layout

A `.trk` file consists of:

1. A **1000-byte binary header** starting with the 6-byte magic `TRACK\0`.
   All multi-byte integer fields are little-endian.
2. For each streamline declared by `n_count`:
   - A 4-byte `n_points` count (i32 LE)
   - `n_points × (3 + n_scalars)` float32 values (position x, y, z followed by
     per-point scalars)
   - `n_properties` float32 values for per-streamline properties

## Coordinate system

Streamlines in memory are Gaia `Polyline<f64>` values in physical RAS+mm
coordinates. On read, the header affine `vox_to_ras` (a 4×4 row-major matrix)
is applied to every voxel coordinate triple to produce physical coordinates.
On write, the inverse affine converts back to voxel space.

## Reading

`TrkTractogram::read` accepts any `impl Read` and returns the parsed header,
streamlines, per-point scalars, and per-streamline properties:

```rust,ignore
use std::fs::File;
use ritk_trk::TrkTractogram;

let mut file = File::open("tracks.trk")?;
let trk = TrkTractogram::read(&mut file)?;
println!("{} streamlines, {} scalars per point",
    trk.streamlines.len(),
    trk.header.n_scalars,
);
```

### Validation

The reader checks:
- Magic bytes (`TRACK\0`)
- Header size equals 1000
- Streamline count is non-negative and bounded (≤ 100M)
- Per-streamline point count is non-negative and bounded (≤ 100K)
- Every coordinate is finite after affine application
- Every polyline is valid Gaia geometry

## Writing

`TrkTractogram::write` serialises the header and streamlines, applying the
inverse of `vox_to_ras` to convert physical coordinates back to voxel space:

```rust,ignore
let mut file = File::create("output.trk")?;
trk.write(&mut file)?;
```

## Header fields

`TrkHeader` is the fixed 1000-byte header. Key fields:

| Field | Type | Meaning |
|---|---|---|
| `dim` | `[i16; 3]` | Voxel grid dimensions |
| `voxel_size` | `[f32; 3]` | Voxel size in mm |
| `origin` | `[f32; 3]` | Origin in mm |
| `n_scalars` | `i16` | Per-point scalar count |
| `scalar_name` | `[u8; 200]` | Space-separated null-terminated scalar names |
| `n_properties` | `i16` | Per-streamline property count |
| `property_name` | `[u8; 200]` | Space-separated null-terminated property names |
| `vox_to_ras` | `[[f32; 4]; 4]` | Row-major 4×4 voxel→RAS+mm affine |
| `voxel_order` | `[u8; 4]` | Voxel ordering convention (e.g. `LPS\0`) |
| `version` | `i32` | Format version (current: 2) |
| `hdr_size` | `i32` | Header size (must be 1000) |

The default header uses an identity `vox_to_ras`, RAS voxel order, version 2,
and `hdr_size` 1000.

## Per-point scalars and properties

When `n_scalars > 0`, per-point scalar values are stored inline after the
position triplet. The `scalar_name` field is a space-separated list of names
(e.g. `"FA MD"`). Scalars are returned as `Vec<Box<[f32]>>` — one flat array
per streamline in `n_points × n_scalars` row-major order.

When `n_properties > 0`, per-streamline properties follow the streamline data
block. Properties are returned as `Vec<Box<[f32]>>` — one array per streamline.

## Error types

`TrkError` covers every failure mode:

| Variant | Condition |
|---|---|
| `InvalidMagic` | First 6 bytes are not `TRACK\0` |
| `InvalidHeaderSize` | `hdr_size` field ≠ 1000 |
| `InvalidStreamlineCount` | `n_count` negative or > 100M |
| `InvalidPointCount` | `n_points` negative or > 100K |
| `UnexpectedEof` | Premature EOF at a known byte offset |
| `NonFiniteCoordinate` | NaN or infinite coordinate after affine |
| `InvalidPolyline` | Gaia rejected the point sequence |

## Usage in the diffusion pipeline

`ritk-tractography` provides `TractographyResult::to_trk`,
`to_trk_header`, and `to_trk_with_scalars` that produce `TrkTractogram`
values directly from tractography output. See the
[Tractography](tractography.md) chapter for the export API.
