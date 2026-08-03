# .tck — MRtrix3 Tractogram Format

`ritk-tck` reads and writes the MRtrix3 `.tck` tractogram format. The format
uses a human-readable text header of `key: value` pairs terminated by the
line `END`, followed by binary streamline data with NaN-delimited streamlines
and an Inf barrier at end-of-file.

## Format layout

A `.tck` file consists of:

1. A **text header** section:
   - The first line is `mrtrix tracks`
   - Key-value pairs in the form `key: value`, one per line
   - The line `END` terminates the header
2. **Binary streamline data** after `END`:
   - Streamline points encoded as 3-tuples of the declared datatype
   - Streamlines separated by a NaN triplet
   - End-of-file signalled by an Inf triplet (or physical EOF)

## Coordinate system

Streamlines are stored in scanner-space millimetre coordinates — the native
`.tck` coordinate system. The optional `transform` header entry maps voxel
space to scanner space as a 4×4 row-major matrix, stored in
`TckHeader::transform` for consumers that need to convert.

## Datatypes

`TckDatatype` supports four binary encodings:

| Variant | Byte width per point |
|---|---|
| `Float32LE` | 12 (3 × f32) |
| `Float32BE` | 12 |
| `Float64LE` | 24 (3 × f64) |
| `Float64BE` | 24 |

The default is `Float32LE`. The datatype is parsed from the `datatype:` header
line and used for both reading and writing.

## Reading

`TckTractogram::read` accepts any `impl Read` and returns the parsed header
and streamlines:

```rust,ignore
use std::fs::File;
use ritk_tck::TckTractogram;

let file = File::open("tracks.tck")?;
let tck = TckTractogram::read(file)?;
println!("{} streamlines, datatype {:?}",
    tck.streamlines.len(),
    tck.header.datatype,
);
```

### Validation

The reader checks:
- The first header line is `mrtrix tracks`
- The `datatype` header value is recognised
- Header lines parse as `key: value` pairs
- Every coordinate is finite (mixed NaN/Inf in a non-delimiter position is
  an error)
- Every polyline is valid Gaia geometry
- The `transform` value (if present) contains exactly 16 floats

The reader gracefully handles EOF mid-streamline (flushing any partial
streamline) and skips empty header lines.

## Writing

`TckTractogram::write` serialises the header and streamlines, using the
declared datatype for all binary encoding:

```rust,ignore
let mut file = File::create("output.tck")?;
tck.write(&mut file)?;
```

The writer emits all well-known header keys in canonical order
(`datatype`, `mrtrix_version`, `file`, `comments`, `count`, `total_count`,
`transform`) before any remaining raw fields, then `END`, then the binary
data.

## Header fields

`TckHeader` carries parsed header metadata with typed accessors alongside
a raw `fields: HashMap<String, String>` for unknown keys:

| Field | Type | Header key |
|---|---|---|
| `count` | `Option<i64>` | `count` |
| `total_count` | `Option<i64>` | `total_count` |
| `datatype` | `TckDatatype` | `datatype` |
| `transform` | `Option<[[f64; 4]; 4]>` | `transform` |
| `mrtrix_version` | `Option<String>` | `mrtrix_version` |
| `file_path` | `Option<String>` | `file` |
| `comments` | `Option<String>` | `comments` |
| `fields` | `HashMap<String, String>` | All key-value pairs |

## Weights sidecar

The `.tck` format does not store per-point scalars inline. MRtrix3 uses a
separate weights file with the same binary layout: one scalar per point
instead of three, with the same NaN delimiter / Inf barrier convention.

`write_tck_weights` writes a weights sidecar file:

```rust,ignore
use ritk_tck::{write_tck_weights, TckDatatype};

let fa_values: Vec<Box<[f32]>> = /* one per streamline, one f32 per point */;
let mut file = File::create("fa.tsf")?;
write_tck_weights(&fa_values, TckDatatype::Float32LE, &mut file)?;
```

`read_tck_weights` reads a weights file back into per-streamline scalar
arrays:

```rust,ignore
use ritk_tck::read_tck_weights;

let file = File::open("fa.tsf")?;
let scalars = read_tck_weights(file)?;
```

## Error types

`TckError` covers every failure mode:

| Variant | Condition |
|---|---|
| `InvalidMagic` | First header line is not `mrtrix tracks` |
| `MalformedHeaderLine` | A line lacks a `key: value` structure |
| `UnknownDatatype` | `datatype` value not one of the four recognised strings |
| `InvalidTransform` | `transform` value does not contain 16 floats |
| `UnexpectedEof` | Premature EOF in binary data |
| `NonFiniteCoordinate` | Mixed NaN/Inf at a non-delimiter position |
| `InvalidPolyline` | Gaia rejected the point sequence |
| `Io` | Underlying I/O error |

## Usage in the diffusion pipeline

`ritk-tractography` provides `TractographyResult::to_tck` and
`to_tck_header` that produce `TckTractogram` values directly from
tractography output, with optional `mrtrix_version`, `comments`, and
`transform` parameters. Use `write_tck_weights` to export per-point
scalars (FA, MD) alongside the tractogram. See the
[Tractography](tractography.md) chapter.
