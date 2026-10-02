# NIfTI Format Boundary

`ritk-nifti` is RITK's native single-source-of-truth implementation for the
single-file NIfTI boundary. It reads NIfTI-1 and NIfTI-2 `.nii` files, detects
gzip-wrapped `.nii.gz` input, and writes either header version explicitly.

## Ownership

`ritk-nifti` owns the NIfTI file reader and writer. `ritk-io::format::nifti`
is a facade re-export. Analyze 7.5 `.hdr`/`.img` pairs belong to
`ritk-analyze`; they are not interpreted as NIfTI by this crate.

The native codec supports:

- three-dimensional scalar images of every fixed-width NIfTI sample type
  (`uint8` through `uint64`, `int8` through `int64`, `float32`, `float64`);
- four-dimensional acquisition series of the same types;
- the `scl_slope`/`scl_inter` rescale from stored to physical values;
- three-dimensional `u32` label maps read from any of those types;
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
one `Image<T, B, 3>` per volume.

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

The example below writes a volume through NIfTI-1 and NIfTI-2, a
three-volume series, and an `int16` volume read as `i16`, widened to `f32`,
and refused as `u8`. It is compiled and run as
`cargo run --example nifti_roundtrip -p ritk-nifti`.

```rust,ignore
{{#include ../../crates/ritk-nifti/examples/nifti_roundtrip.rs}}
```

`read_nifti_series_from_bytes` provides the same decoding contract for an
in-memory `.nii` or `.nii.gz` payload.

## Sample Types and Conversion

The `datatype` field names how each sample is stored. Every reader is
generic over the sample type `T` the caller asks for, and takes a conversion
policy from `ritk_codecs::sample` as a zero-sized value (ADR 0053):

| Policy | Accepts | Effect |
|---|---|---|
| `Exact` | the stored type, or a type it widens to without loss | no value changes; any other request is an error that names both types |
| `Cast` | every type | the primitive `as` cast, with a warning when the stored type does not widen |

The widening pairs are the ones the standard library implements `From`
for: `int16` widens to `int32`, `int64`, `float32`, and `float64`; `uint32`
widens to `float64` but not to `float32`, which cannot hold every `uint32`.

A nonzero, finite `scl_slope` declares the rescale
`physical = scl_slope · stored + scl_inter`. The readers apply it in `T`'s
own arithmetic after conversion, so a CT stored as `int16` reads directly as
Hounsfield units in `f32`. A rescale into an integer `T` has no faithful
result and is an error; `read_nifti_stored` and `read_nifti_series_stored`
return the stored samples together with the `Rescale`, leaving the mapping to
the caller. A zero or non-finite slope means no rescale; a valid slope with a
non-finite intercept is an error. A coefficient outside `T`'s range, or a
nonzero slope that rounds to zero in `T`, is the error `RescaleOutOfRange`
and leaves the samples unchanged.

The writers emit the `datatype` code and `bitpix` of the image's `T` and no
rescale, so a written file reads back in its own type unchanged.

Label maps read from any stored type: an integer voxel must fit `u32`, and a
floating-point voxel must hold a whole number in `0..=u32::MAX`. A negative,
fractional, or out-of-range voxel is an error rather than a rounded or
clamped label.

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
