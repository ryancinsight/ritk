# Example: DICOM to NIfTI Conversion

Reads a DICOM series through RITK and writes a NIfTI volume.

## Source

`crates/ritk-io/examples/dicom_to_nifti.rs`

## Description

The optional `series_uid` selects a series from a directory containing more
than one image series. Without it, RITK accepts only an unambiguous directory.
Both pixel decoding and NIfTI output use the RITK I/O APIs.

## Usage

```bash
cargo run --example dicom_to_nifti -- <input_dicom_series_dir> <output_nifti> [series_uid]
```

## Verification

- Reads one selected DICOM image series
- Writes a NIfTI file with preserved image geometry
- Rejects ambiguous series directories unless a UID is supplied
