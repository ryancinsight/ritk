# NIfTI Format Boundary

`ritk-nifti` is RITK's native single-source-of-truth implementation for the
single-file NIfTI boundary. It reads NIfTI-1 and NIfTI-2 `.nii` files, detects
gzip-wrapped `.nii.gz` input, and writes either header version explicitly.

## Ownership

`ritk-nifti` owns the NIfTI file reader and writer. `ritk-io::format::nifti`
is a facade re-export. Analyze 7.5 `.hdr`/`.img` pairs belong to
`ritk-analyze`; they are not interpreted as NIfTI by this crate.

The image convenience API reads and writes three-dimensional `f32` images,
four-dimensional `f32` acquisition series, and `u32` label maps. These APIs
project voxel values to their image surface types. Use `NiftiDocument` when the
stored datatype and exact sample bit patterns must be retained.

`NiftiDocument::from_stored_series` constructs a document from RITK's
`StoredSeries` without an intermediate file. It writes exact payload bits for
the supported scalar types `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`,
`i64`, `f32`, and `f64`. It preserves a single-volume axis and an ordered-list
axis with at least two volumes. A singleton list is rejected because a rank-3
header would erase its fourth axis; unspecified and diffusion axes return
typed capability losses. NIfTI-1
stores dimensions as signed 16-bit integers, so every positive axis and volume
count is bounded by 32,767, as specified by the official [NIfTI-1 dimension
field reference](https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/dim.html/document_view.html)
and [data-format FAQ](https://nifti.nimh.nih.gov/nifti-1/documentation/faq.html).
It represents spatial and scaling fields as 32-bit floats and rejects values
that would need rounding. NIfTI-2 stores dimensions as signed 64-bit integers
([NIfTI-2 format overview](https://nifti.nimh.nih.gov/nifti-2/index_html/view.html))
and represents spatial and scaling fields as 64-bit floats. The converter
rejects values outside the selected header representation before producing a
document.

`NiftiDocument` is the lossless transport surface. It retains the complete uncompressed single-file
stream, including unprojected header fields, the extension indicator and blocks, and exact sample bits.
Writing to `.nii` reproduces those bytes; writing to `.nii.gz` changes only the gzip
framing. The gzip reader drains the stream through its checksum trailer before acceptance.

The typed header view exposes both transform codes and classifies the active forms as qform-only,
sform-only, compatible in handedness, or conflicting in handedness. Both forms remain in the document
when they differ: qform can describe scanner coordinates while sform describes a standard space. A
handedness conflict is reported explicitly because the NIfTI-1 specification
states that operations on such an image are unspecified. See the official
[NIfTI-1 FAQ, questions 19 and 21](https://nifti.nimh.nih.gov/nifti-1/documentation/faq.html).

`transcode_nifti_document` performs `.nii`/`.nii.gz` framing conversion inside
RITK. It validates and, for gzip output, finishes compression before opening
the destination, so format or compression errors leave an existing output
unchanged.

The official [NIfTI-1 field reference](https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/index.html)
defines the scalar field widths, and the [NIfTI-1 FAQ](https://nifti.nimh.nih.gov/nifti-1/documentation/faq.html)
defines zero `scl_slope` as the scaling-disabled value.

## Spatial Contract

NIfTI file-axis RAS maps to RITK `[depth, row, col]` through the format
boundary. The reader constructs each image directly as `[nz, ny, nx]` from
X-fastest NIfTI raw bytes; the writer emits RITK ZYX flat data in that file
order.

## Affine Conversion

- Reader: file affine columns `[x,y,z]` become internal metadata columns
  `[depth,row,col] = [z,y,x]`.
- Writer: sform columns are emitted as `[internal_col, internal_row, internal_depth]`.

RITK physical metadata uses LPS coordinates, while NIfTI affines use RAS.
The boundary performs the LPS/RAS sign conversion; callers must not pre-flip
their images.

## Acquisition Series

A repeated acquisition is represented by a NIfTI rank-4 image:

```text
dim[0] = 4
dim[1..=3] = [nx, ny, nz]
dim[4] = number of volumes
```

The fourth axis can represent diffusion gradient directions, functional
timepoints, or another repeated measurement. NIfTI stores this axis slowest,
so the complete voxels for volume 0 are followed by volume 1, then volume 2,
and so on. `read_nifti_series` preserves that acquisition order and returns
one `Image<f32, B, 3>` per volume.

Every volume shares one shape and one physical grid because a NIfTI series
carries one spatial transform. The series writers therefore reject an empty
series or any volume whose shape, origin, spacing, or direction differs from
volume 0. This prevents one header from silently describing only part of the
written data.

The stored-series constructor also requires one sample type and one global
linear calibration across the series. Identity calibration disables NIfTI
scaling; a zero-slope source mapping, nonlinear modality lookup, or differing
per-frame mappings is rejected because it cannot be represented by the NIfTI
header. Diffusion metadata and non-Cartesian coordinates are reported through
the conversion capability error rather than discarded. The header stores one
sform in RAS coordinates; RITK's LPS origin, spacing, direction, and
`[depth,row,column]` axes are mapped at the codec boundary.

### Rank behavior

The single-volume and series APIs are deliberately asymmetric:

| File | `read_nifti` | `read_nifti_series` |
|---|---|---|
| Rank 3 | returns one image | returns a one-image vector |
| Rank 4 | rejects the file and reports its volume count | returns every volume in order |

Returning volume 0 through the single-image API would discard the rest of an
acquisition while reporting success. Conversely, a rank-3 image is a valid
series of one and needs no caller-side rank branch.

Writing follows the same canonical representation. A one-image slice passed
to `write_nifti_series` is written as rank 3 and remains readable by
`read_nifti`. Two or more images are written as rank 4 with their count in
`dim[4]`.

### Public API

```rust,ignore
use coeus_core::SequentialBackend;
use ritk_nifti::{
    read_nifti_series, write_nifti2_series, write_nifti_series,
};

let backend = SequentialBackend;

// All images must have the same shape and physical metadata.
write_nifti_series("diffusion.nii.gz", &volumes, &backend)?;
let decoded = read_nifti_series("diffusion.nii.gz", &backend)?;

// Select NIfTI-2 explicitly when its wider header fields are required.
write_nifti2_series("diffusion-nifti2.nii", &decoded, &backend)?;
```

`read_nifti_series_from_bytes` provides the same decoding contract for an
in-memory `.nii` or `.nii.gz` payload.

## Validation and Failure Semantics

The boundary validates dimensions, voxel byte ranges, datatype widths,
spatial metadata, and gzip expansion limits before constructing images. A
declared payload that ends after only some volumes is an error; the reader
does not return the complete prefix as a partial series.

The writer reports the zero-based position and mismatched grid property when
a volume cannot share the first volume's spatial transform. Choose the
single-volume API for one image and the series API whenever the input may
contain repeated acquisitions.

## Invariant

NIfTI parser/writer dependency changes stay behind `ritk-nifti`; callers
in `ritk-io`, CLI, and viewer code consume the same authoritative API.
