# NIfTI Format Boundary

`ritk-nifti` is RITK's native single-source-of-truth implementation for the
single-file NIfTI boundary. It reads NIfTI-1 and NIfTI-2 `.nii` files, detects
gzip-wrapped `.nii.gz` input, and writes either header version explicitly.

## Ownership

`ritk-nifti` owns the NIfTI file reader and writer. `ritk-io::format::nifti`
is a facade re-export. Analyze 7.5 `.hdr`/`.img` pairs belong to
`ritk-analyze`; they are not interpreted as NIfTI by this crate.

The native codec supports:

- three-dimensional `f32` scalar images;
- four-dimensional `f32` acquisition series;
- three-dimensional `u32` label maps;
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
