# Example: Geometry Validation

Verify ritk's NIfTI import + index-to-world against SimpleITK ground truth.

## Source

`crates/ritk-registration/examples/geometry_check.rs`

## Description

Tests that the NIfTI import pipeline correctly converts file-axis metadata
to internal spatial coordinates. Prints geometry and index-to-world for
fixed voxel indices; compare output to sitk reference.

## Usage

```bash
cargo run --example geometry_check
```

## Verification

- Loads NIfTI file with known affine
- Computes index-to-world for corner/mid-volume voxels
- Compares against SimpleITK ground truth
- Exercises the `ritk-io::format::nifti` spatial boundary
