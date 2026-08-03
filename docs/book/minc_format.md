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
`image` dataset and writes one contiguous little-endian `f32` dataset. The
reader supports global and first-spatial-axis per-slice integer scaling. It does
the same for boolean storage, treating false and true as stored codes 0 and 1.
It does not yet read chunked/compressed datasets, expose arbitrary metadata
under `info`, or write multiresolution levels. Those cases return an error where
they are detectable.

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
`f32`, and `f64` payloads and returns `f32`. The current writer accepts
`Image<f32, B, 3>` and preserves the little-endian IEEE-754 bits exactly.

For integer and boolean image data, the
[MINC pixel-conversion specification](https://www.bic.mni.mcgill.ca/software/minc/prog_guide/node19.html)
maps each stored value \(v\) to a real intensity \(r\):

\[
r = r_{\min} +
    (v-v_{\min})\frac{r_{\max}-r_{\min}}{v_{\max}-v_{\min}}.
\]

Here \([v_{\min},v_{\max}]\) is the image dataset's `valid_range`, whose
[endpoint order is insignificant](https://www.bic.mni.mcgill.ca/software/minc/minc2_format/node15.html).
If it is absent, RITK uses the complete stored integer range. The real range
\([r_{\min},r_{\max}]\) comes from `image-min` and `image-max`. Those datasets
may be scalar for one global mapping or contain one value per slice along the
first spatial image axis. If both are absent, the MINC default real range is
`[0, 1]`. Equal `valid_range` endpoints are malformed because the conversion
denominator is zero, so the reader rejects them. For boolean storage, the
default valid range is `[0, 1]`; false and true then follow the same mapping.

Values outside `valid_range` denote missing or uninitialized data. RITK's
`Image<f32, B, 3>` has no missing-value mask, so the reader returns a contextual
error naming the first invalid voxel instead of silently inventing a value.
Floating-point image datasets bypass `image-min`/`image-max` scaling, as the
MINC conversion contract requires. Integer values beyond binary64's exact
integer range and finite `f64` image values can still round under the explicit
`f32` output contract.

## Bounded reading and writing

The reader validates exact dataset and dimension shapes and calculates element,
slice, and byte counts with checked arithmetic. It reads at most 8 KiB of raw
voxel bytes at a time and decodes directly into the returned `f32` volume. A
hostile shape that claims more voxels than the file backs therefore fails on
the first unbacked chunk without reserving a second volume-sized byte buffer.
Scaling is applied while each chunk enters the final output.

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
overflow, truncated voxel data, malformed scaling ranges, incomplete
`image-min`/`image-max` pairs, out-of-range stored values, and
image-construction failure. Writing reports empty or unrepresentable axes,
storage/shape disagreement, invalid physical metadata, allocation failure, and
positioned-write failure. Logical preflight errors occur before the output path
is created.

The [MINC2 round-trip example](examples/minc_roundtrip.md) makes the visual and
numerical comparison explicit.
