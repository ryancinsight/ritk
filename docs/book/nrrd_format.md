# NRRD Format Boundary

Single source of truth for NRRD file I/O.

## Ownership

`ritk-nrrd` owns the NRRD file reader and writer. `ritk-io::format::nrrd`
is a facade re-export.

## Spatial Contract

NRRD file-axis `[x,y,z]` maps to RITK `[depth,row,col]` via `crates/ritk-nrrd/src/spatial.rs`.
The reader constructs the tensor directly as `[nz,ny,nx]` from X-fastest NRRD
raw bytes; the writer emits RITK ZYX flat data directly.

## Direction Vectors

- Reader: NRRD `space directions` vectors `[x,y,z]` become internal metadata
  columns `[depth,row,col] = [z,y,x]`.
- Writer: NRRD `space directions` are generated from internal columns
  `[col,row,depth]`.

A rank-2 NRRD carries two-component direction vectors and origin coordinates.
The reader validates those planar values before promoting the image to a
degenerate `[1,Y,X]` volume with unit through-plane spacing and zero
through-plane origin. Rank-3 and rank-4 files continue through the spatial and
acquisition-axis parser, so two-component vectors are never interpreted as
truncated 3-D metadata.

## Sample Types

The reader decodes the type the `type` field names, then converts to the
caller's sample type `T` under a conversion policy (`Exact` or `Cast`, from
`ritk-codecs`); the writer stores `T` itself. All ten numeric NRRD types are
supported, each under every name the NRRD file format specification lists:

| Stored type | `type` names (first is the one the writer emits) |
| --- | --- |
| `i8` | `signed char`, `int8`, `int8_t` |
| `u8` | `unsigned char`, `uchar`, `uint8`, `uint8_t` |
| `i16` | `short`, `short int`, `signed short`, `signed short int`, `int16`, `int16_t` |
| `u16` | `unsigned short`, `ushort`, `unsigned short int`, `uint16`, `uint16_t` |
| `i32` | `int`, `signed int`, `int32`, `int32_t` |
| `u32` | `unsigned int`, `uint`, `uint32`, `uint32_t` |
| `i64` | `long long int`, `longlong`, `long long`, `signed long long`, `signed long long int`, `int64`, `int64_t` |
| `u64` | `unsigned long long int`, `ulonglong`, `unsigned long long`, `uint64`, `uint64_t` |
| `f32` | `float` |
| `f64` | `double` |

Names compare case-insensitively with whitespace collapsed. `block` stores
untyped records and is rejected, as is any name outside the table. The
`endian` field is `big` or `little`; any other value is an error, because a
payload decoded in a guessed byte order is wrong data. The field is required
for every type wider than one byte, as the format specification requires for
raw and gzip payloads, and a file without it is refused; a one-byte type has no
byte order and reads without the field.
`raw` and `gzip` payloads, inline or in a detached `data file`, read in either
byte order; the writer emits `raw` little-endian. The payload decodes as a
stream of exactly the declared sample count, so a header that overstates its
sizes fails on the short stream instead of allocating them.

`ritk-io` reads through `Cast` at its `f32` surfaces, so an `int`,
`unsigned int`, `long long int`, `unsigned long long int`, or `double` file
reads there with a warning where `f32` cannot hold every value.

## Invariant

NRRD parser/writer dependency changes stay behind `ritk-nrrd`; callers
in `ritk-io`, CLI, and viewer code consume the same authoritative API.

## Diffusion gradient metadata

`read_nrrd_gradient_scheme` implements the NA-MIC DWI convention. One nominal
`DWMRI_b-value` is combined with each `DWMRI_gradient_XXXX` squared norm to
recover the per-volume effective b-value. The measurement frame maps gradients
to world coordinates; RAS world coordinates are converted once to RITK LPS.
Missing indices, non-finite values, `DWMRI_NEX`, and B-matrix encodings fail
explicitly rather than being guessed.
