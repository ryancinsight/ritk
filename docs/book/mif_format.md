# MRtrix `.mif` Format Boundary

`.mif` is MRtrix3's container format: a plain-text header terminated by a bare
`END` line, followed directly by raw binary voxel data in the same file
(inline), or in a sidecar `.mif.dat` named by the header's `file:` key
(detached). RITK exposes both layouts through the `ritk-mif` crate:

- `read_mif` reads one-volume `.mif` and rejects multi-frame input;
- `read_mif_series` reads every frame of an acquisition or diffusion series;
- `write_mif` writes one volume; `write_mif_series` writes an interleaved
  multi-frame file;
- `MifReader` is a thin adapter whose `read` method receives the backend;
- `MifWriter` retains a backend for repeated operations.

MRtrix3 documents the format in its
[image format reference](https://mrtrix.readthedocs.io/en/latest/getting_started/image_data.html).

## On-disk organization

```text
byte 0                                                        end of file
┌────────────── text header, `END`-terminated ──────────────┬──────────────┐
│ mrtrix image: version 3.0, dim, vox, layout, datatype, …  │ voxel bytes  │
└───────────────────────────────────────────────────────────┴──────────────┘
                                                               x → y → z → frame
```

Header values are single-line (`dim: 128 128 60`) or multi-line blocks
(`transform:` followed by four rows). A trailing backslash continues a line;
`#` starts a comment.

| Key | Meaning |
| --- | --- |
| `mrtrix image` | Magic line, carrying the format version. |
| `dim` | Extents in file `[x, y, z]` order, optionally a fourth frame axis. |
| `vox` | Voxel sizes in file `[x, y, z]` order, in millimetres. |
| `layout` | Axis strides, e.g. `+0,+1,+2`; absent means contiguous. |
| `datatype` | Sample type, optionally with a `LE`/`BE` suffix. |
| `transform` | 4×4 row-major affine mapping voxel `[x,y,z]` to scanner millimetres. |
| `file` | Detached-payload hint: a relative path and an offset. |
| `DW_scheme` | Diffusion gradient table (block). |

The reader accepts `float32`, `float64`, `int32`, `uint32`, `int16`, `uint16`,
`int8`, and `uint8`. The writer emits `float32`. A `BE` suffix selects
big-endian; the default is little-endian. Every axis must span at least one
voxel — a zero extent is rejected, because zero voxels would also make a
truncated payload satisfy the length check.

## Spatial convention

RITK tensors are `[depth, row, col]` = `[z, y, x]`, while the `.mif` file names
`[x, y, z]`. The raw payload is already in RITK's flat order — X varies fastest
in both — so no data permutation is needed; only the *metadata* order differs:

- `transform` is decomposed into RITK origin, spacing, and direction; the
  writer reorders its columns from internal `[depth, row, col]` to file
  `[x, y, z]`.
- A header with no `transform` is axis-aligned identity, and its `vox:` triple —
  file `[x, y, z]` — is reversed into `[Δdepth, Δrow, Δcol]`. Reading the three
  components index-for-index would transpose the spacing against the `dim`
  line, which the same file decodes as `[nz, ny, nx]`.

A round-trip test cannot catch that transposition: the writer always emits a
`transform`, so the transform-less branch is reached only by files this crate
did not write, and a writer/reader that agree on a permutation round-trip
exactly. The axis order is therefore pinned by a hand-built-header oracle in
`ritk-mif`'s reader tests, not by the round-trip suite.

## `ritk-io` route

`ritk-io::format::mif` re-exports the codec and defines the local
`ImageReader`/`ImageWriter` adapters, so `ImageFormat::Mif` and
`read_image_native`/`write_image_native` reach `.mif` without a consumer
depending on `ritk-mif` directly.
