# Example: Geometry Validation

Verify RITK's NIfTI import and index-to-world mapping before registration.

## Source

`crates/ritk-registration/examples/geometry_check.rs`

## Description

The example accepts a CT and MR NIfTI path, prints each image's shape,
spacing, origin, direction, and several index-to-world samples. The output
is the geometry contract to check before a registration metric or resampler
combines the images.

## Usage

```bash
cargo run -p ritk-registration --example geometry_check -- \
  /data/ct.nii.gz /data/mr.nii.gz
```

The example does not assume a particular dataset or filesystem layout. For
the repository's downloaded fixture, replace the two paths with the CT/MR
files available in your local data directory.

## Verification

- Loads both NIfTI files through `ritk_io::format::nifti::native::NiftiReader`
- Computes index-to-world for corner and interior voxels that fit each volume
- Makes axis order explicit: RITK stores the native volume as `[z, y, x]`
- Exercises the `ritk-io::format::nifti` spatial boundary without private paths
