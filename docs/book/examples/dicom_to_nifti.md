# Example: DICOM to NIfTI Conversion

Analyze 7.5 to NIfTI converter example.

## Source

`crates/ritk-io/examples/dicom_to_nifti.rs`

## Description

This example loads an Analyze 7.5 image pair and saves it as a NIfTI file.
It exercises the `ritk-io` DICOM reader and NIfTI writer boundaries.

## Usage

```bash
cargo run --example dicom_to_nifti -- <input_dir> <output_nifti>
```

## Verification

- Reads a scalar Analyze 7.5 image
- Writes a NIfTI file with correct affine metadata
- Round-trips through `ritk-image` boundary types
