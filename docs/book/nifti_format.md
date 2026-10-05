# NIfTI Format Boundary

`ritk-nifti` implements RITK's single-file NIfTI reader and writer. It reads
NIfTI-1 and NIfTI-2 `.nii` files, detects gzip-wrapped `.nii.gz` input, and
writes either header version explicitly.

## Ownership

`ritk-nifti` owns the NIfTI file reader and writer. `ritk-io::format::nifti`
is a facade re-export. Analyze 7.5 `.hdr`/`.img` pairs belong to
`ritk-analyze`; they are not interpreted as NIfTI by this crate.

The codec supports:

- three-dimensional `f32` scalar images;
- four-dimensional `f32` acquisition series;
- three-dimensional `u32` label maps;
- three-dimensional typed stored volumes covering all ten RITK scalar types;
- NIfTI sform and qform spatial metadata; and
- NIfTI-1 and NIfTI-2 single-file streams, compressed or uncompressed.

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

## Exact Stored Samples

The legacy `read_nifti` API returns `f32` images. Use the stored-volume API
when the on-disk sample representation must remain exact. It reads only one
rank-three volume and retains the sample type and every stored bit pattern;
calibration remains separate from the sample buffer.

| NIfTI scalar datatype | RITK stored type |
|---|---|
| `DT_UINT8` | `u8` |
| `DT_INT8` | `i8` |
| `DT_UINT16` | `u16` |
| `DT_INT16` | `i16` |
| `DT_UINT32` | `u32` |
| `DT_INT32` | `i32` |
| `DT_UINT64` | `u64` |
| `DT_INT64` | `i64` |
| `DT_FLOAT32` | `f32` |
| `DT_FLOAT64` | `f64` |

The reader accepts single-file `.nii` and gzip-compressed `.nii.gz`. It checks
the encoded and decoded byte limits from the header before allocating the
sample buffer. NIfTI extension records are rejected because
`StoredVolume` has no extension-metadata field; returning the voxel values
while silently dropping an extension would not preserve the input.

`write_nifti_stored` writes NIfTI-2 so the affine and linear calibration fields
retain their `f64` values. `.nii.gz` selects gzip output. The writer preserves
Cartesian LPS-millimeter geometry, sample type, and sample bits. NIfTI provides
one global linear intensity transform: identity and nonzero linear calibration
are representable, and identical per-frame transforms collapse to that global
transform. Modality lookup tables, varying per-frame transforms, zero-slope
linear calibration, and non-Cartesian coordinate maps are rejected before the
destination is created or truncated. A zero `scl_slope` disables NIfTI scaling,
so it cannot represent a non-identity zero-slope transform.

```rust,ignore
{{#include ../../crates/ritk-nifti/examples/nifti_stored_roundtrip.rs}}
```

Run the same source used by this page with an input NIfTI volume and an output
path:

```bash
cargo run -p ritk-nifti --example nifti_stored_roundtrip -- scan.nii.gz copy.nii.gz
```

NIfTI datatype codes follow the [NIfTI-1 specification](https://nifti.nimh.nih.gov/dfwg/presentations/nifti1_cox.pdf/download).
NIfTI-2 widens the header fields for 64-bit storage and addressing, as defined
by the [NIfTI-2 format specification](https://nifti.nimh.nih.gov/nifti-2/index_html/view.html).
The `scl_slope` and `scl_inter` behavior follows the official
[data-scaling description](https://nifti.nimh.nih.gov/dfwg/presentations/nifti-1-rationale.html).

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
