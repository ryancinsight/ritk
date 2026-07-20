# Example: DICOM Dump Utility

Dump DICOM file metadata and pixel values.

## Source

`crates/ritk-io/examples/dump_dicom.rs`

## Description

Utility to inspect DICOM file metadata, pixel value ranges, and spatial
transformation data. Exercises the `ritk-dicom` parser boundary.

## Usage

```bash
cargo run --example dump_dicom -- <dicom_file> [--json]
```

## Verification

- Parses DICOM Part 10 file
- Extracts patient/study/series metadata
- Reports pixel value statistics (min, max, mean)
