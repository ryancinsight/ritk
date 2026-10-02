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
`image` dataset and writes one contiguous little-endian dataset of the image's
own sample type. The reader supports global and first-spatial-axis per-slice
integer scaling.
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

The reader accepts contiguous HDF5 signed and unsigned 8-, 16-, 32-,
and 64-bit integer, `f32`, and `f64` payloads in either byte order. It decodes
them in the stored type and converts to the caller's sample type `T` under a
conversion policy: `Exact` refuses any conversion that could change a value
(for example `i32` samples read as `f32`), and `Cast` converts and logs a
warning. The writer stores `T` itself, little-endian: `u8`, `i8`, `u16`, `i16`,
`u32`, `i32`, `f32`, or `f64`. MINC2 has no 64-bit integer voxel type, so the
writer refuses `u64` and `i64` before creating a file. A floating-point image
written and read in the same type preserves the IEEE-754 bits exactly.

For integer image data, the
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
denominator is zero, so the reader rejects them. RITK reads a `u8` dataset
whose `valid_range` is `[0, 1]` and which has no `image-min`/`image-max`, such
as a binary mask, by this rule: codes 0 and 1 map through the default real
range `[0, 1]` to 0 and 1, and any other code is outside `valid_range`.

The conversion is linear in the stored value, so each slice is one
`RealValueMap` \(r = (v-v_{\min})\,a + r_{\min}\) with
\(a=(r_{\max}-r_{\min})/(v_{\max}-v_{\min})\). The offset \(v-v_{\min}\) is
applied before the scale, never folded into the intercept: for a stored integer
it is exact, whereas the folded intercept \(r_{\min}-v_{\min}a\) cancels
catastrophically when \(v_{\min}\) is far from zero (`valid_range`
\([60000, 65535]\), \(v=60001\), real range \([0,1]\) maps to
\(1.8067\times10^{-4}\), where the folded form in `f32` gives
\(1.8024\times10^{-4}\)). A slice with \(r_{\min}=r_{\max}\) is uniform:
\(a=0\). `read_minc` evaluates the map in the arithmetic of `T`, so `T` must be
a floating-point type unless the map is the identity (\(a=1\) and
\(r_{\min}=v_{\min}\)). `read_minc_stored` returns the stored samples together
with one `RealValueMap` per slice, unapplied, and is the only way to read an
image with a non-identity map into an integer type. The writer stores an
integer image with `image-min` and `image-max` equal to its type's range, which
with the default `valid_range` is the identity map, so its output reads back
unchanged in `T`.

Values outside `valid_range` denote missing or uninitialized data. RITK's
`Image<T, B, 3>` has no missing-value mask, so the reader returns a contextual
error naming the first invalid voxel instead of silently inventing a value.
Floating-point image datasets bypass `image-min`/`image-max` scaling, as the
MINC conversion contract requires. The map's coefficients are `f64`
and convert to `T` once each. In `T`'s arithmetic the offset is exact for a
stored integer `T` represents exactly, the multiply rounds once, and the add
rounds once, so a mapped value differs from libminc's `f64` evaluation, rounded
to `T`, by at most \(10u(|(v-v_{\min})a|+|r_{\min}|)\) with \(u\) the unit
roundoff of `T`. The map runs in `T`, never in a wider type, so the
intermediate \((v-v_{\min})a\) must be finite in `T` as well as the result:
the reader returns an error when either leaves `T`'s finite range, even if the
mapped value itself would fit (an `f32` read of a file whose intermediate is
\(6\times10^{38}\) is refused, while an `f64` read holds it). The error
suggests reading into a wider floating-point type.

## Bounded reading and writing

The reader validates exact dataset and dimension shapes and calculates element,
slice, and byte counts with checked arithmetic. It reads at most 16 KiB of raw
voxel bytes at a time into a buffer of the stored type that grows only by the
samples already read. A hostile shape that claims more voxels than the file
backs therefore fails on the first unbacked chunk without reserving the claimed
volume. The per-slice maps expand from the scalar or per-slice ranges the file
holds only after the payload has been read, so a header claiming more slices
than the file backs allocates nothing proportional to the claim. Conversion and
scaling run once over the decoded samples, in `T`.

Before file creation, the writer checks the voxel product, the format's `i32`
dimension limit, payload bytes, storage length, and physical metadata. It then
converts at most 2,048 voxels at a time into one heap-allocated
little-endian scratch buffer of 2,048 times the sample width bytes (8 KiB for
a 4-byte sample, 16 KiB for an 8-byte one). Writer-owned scratch is therefore
constant with volume size; it no longer duplicates the complete volume as
bytes.

```rust,ignore
use coeus_core::SequentialBackend;
use ritk_codecs::sample::Exact;
use ritk_minc::{read_minc, read_minc_stored, write_minc};

let backend = SequentialBackend;
// Real intensities in f32; an i16 file reads exactly.
let image = read_minc::<f32, _, _, _>("brain.mnc", &backend, Exact)?;
write_minc(&image, "copy.mnc", &backend)?;
// The stored samples and one linear map per slice, unapplied.
let (stored, maps) = read_minc_stored::<i16, _, _, _>("brain.mnc", &backend, Exact)?;
```

## Failure behavior

Reading reports malformed HDF5 structure, absent image or dimensions, invalid
dimension attributes, unsupported storage layout or scalar type, byte-count
overflow, truncated voxel data, malformed scaling ranges, incomplete
`image-min`/`image-max` pairs, out-of-range stored values, and
image-construction failure. Writing reports empty or unrepresentable axes,
storage/shape disagreement, a sample type MINC2 cannot store (`u64`, `i64`),
invalid physical metadata, allocation failure, and
positioned-write failure. Logical preflight errors occur before the output path
is created.

The [MINC2 round-trip example](examples/minc_roundtrip.md) makes the visual and
numerical comparison explicit.
