# MINC2 Format Boundary

MINC2 stores medical images in an HDF5 hierarchy. The format separates voxel
arrays from named dimensions and supporting scan metadata. The
[MINC2 format reference](https://www.bic.mni.mcgill.ca/software/minc/minc2_format/)
defines the hierarchy and attributes; the later
[MINC2 design paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC4980430/)
explains its HDF5 organization, coordinate model, scaling, chunking, and
multiresolution design.

## Hierarchy and RITK's current profile

A full MINC2 namespace can contain:

```text
/minc-2.0
├── dimensions
│   ├── xspace
│   ├── yspace
│   └── zspace
├── image
│   └── 0
│       ├── image
│       ├── image-min
│       └── image-max
└── info
```

The standard's
[minimal-file rule](https://www.bic.mni.mcgill.ca/software/minc/minc2_format/node26.html)
requires the image and its dimensions; the three principal groups form the
recommended framework. RITK currently reads one three-dimensional contiguous
`image` dataset and writes one contiguous little-endian `f32` dataset. It does
not yet read chunked/compressed datasets, apply `image-min`/`image-max`
real-value scaling, expose arbitrary metadata under `info`, or write
multiresolution levels. Those cases return an error where they are detectable;
scaled integer data is not yet a supported quantitative interchange path.

This restricted profile is useful for exact RITK round trips. It is not a claim
that every MINC2 variant is supported. Validate representative foreign files
before using the reader in a quantitative or clinical pipeline.

## Dimensions and physical coordinates

MINC names spatial axes `xspace`, `yspace`, and `zspace`. The `dimorder`
attribute on the image dataset gives the slowest-to-fastest array order. The
[dimension attribute specification](https://www.bic.mni.mcgill.ca/software/minc/minc2_format/node19.html)
defines:

| Attribute | Meaning in RITK |
|---|---|
| `length` | positive axis extent |
| `start` | world coordinate at index zero |
| `step` | signed sampling interval |
| `direction_cosines` | normalized physical direction of the axis |

RITK's three-dimensional shape is `[z, y, x]`; x remains the fastest-changing
voxel index. With ordered dimension records \(d_0,d_1,d_2\), the physical point
for image index \(i\) is

\[
p(i) = o + D\,\operatorname{diag}(s)\,i,
\]

where \(o_k=\text{start}(d_k)\), \(s_k=|\text{step}(d_k)|\), and column \(k\)
of \(D\) is the corresponding direction cosine with the step sign absorbed.
The writer requires finite origins, positive finite spacing, and orthonormal
direction columns before creating a file.

## Scalar conversion

The reader accepts contiguous HDF5 boolean, signed integer, unsigned integer,
`f32`, and `f64` payloads and returns `f32`. Integer values beyond binary32's
exact integer range and finite `f64` values can round under this explicit
output contract. The current writer accepts `Image<f32, B, 3>` and preserves
the little-endian IEEE-754 bits exactly.

Because RITK does not yet apply MINC real-value scaling, a foreign integer file
that uses `valid_range`, `image-min`, and `image-max` must not be interpreted as
calibrated intensity. The
[image attribute specification](https://www.bic.mni.mcgill.ca/software/minc/minc2_format/node15.html)
describes that scaling metadata.

## Bounded reading and writing

The reader calculates element and byte counts with checked arithmetic. It reads
the contiguous dataset through a bounded growth helper, so a hostile shape that
claims more voxels than the file backs fails on I/O without reserving the full
declared payload first.

Before file creation, the writer checks the voxel product, the format's `i32`
dimension limit, payload bytes, storage length, and physical metadata. It then
converts at most 2,048 voxels at a time into one stack-backed 8 KiB
little-endian scratch buffer. Writer-owned scratch is therefore constant with
volume size; it no longer duplicates the complete volume as bytes.

```rust,ignore
use coeus_core::SequentialBackend;
use ritk_minc::{read_minc, write_minc};

let backend = SequentialBackend;
let image = read_minc("brain.mnc", &backend)?;
write_minc(&image, "copy.mnc", &backend)?;
```

## Failure behavior

Reading reports malformed HDF5 structure, absent image or dimensions, invalid
dimension attributes, unsupported storage layout or scalar type, byte-count
overflow, truncated voxel data, and image-construction failure. Writing reports
empty or unrepresentable axes, storage/shape disagreement, invalid physical
metadata, allocation failure, and positioned-write failure. Logical preflight
errors occur before the output path is created.

The [MINC2 round-trip example](examples/minc_roundtrip.md) makes the visual and
numerical comparison explicit.
