# ritk-io

Unified medical image I/O for [RITK](https://github.com/ryancinsight/ritk).

Owns cross-format dispatch and the `ImageReader` / `ImageWriter` contracts; the
byte-level parsing lives in the per-format crates.

| Format | Read | Write |
|---|---|---|
| DICOM (series) | yes | yes |
| NIfTI (`.nii` / `.nii.gz`) | yes | yes |
| MetaImage (`.mha` / `.mhd`) | yes | yes |
| NRRD | yes | yes |
| PNG | yes | yes |
| TIFF / BigTIFF | yes | yes |
| MGH / MGZ (FreeSurfer) | yes | yes |
| Analyze 7.5 | yes | yes |
| MINC2 | yes | yes |
| VTK legacy structured points | yes | yes |
| JPEG | yes | 2-D grayscale only |

`read_image_native` and `write_image_native` select the format by path and
content. The crate also ships a DICOMweb client (QIDO / STOW) and a PS 3.15
Annex E de-identification toolset with an export-time metadata integrity gate —
see `examples/anonymize_pacs_export.rs`.

## Usage

```toml
[dependencies]
ritk-io = "0.3.0"
```

DICOM opening preserves acquisition identity: `scan_dicom_path` accepts a
selected instance, directory, or explicit DICOMDIR; `scan_dicom_files` scans
an exact discovered member set. Feed the resulting descriptor to
`load_dicom_from_series` to decode pixels and retain metadata. Multiple
SeriesInstanceUID values require explicit selection; missing or invalid image
UIDs and inconsistent dimensions fail before geometry assembly. Named byte
batches use the same single-series contract. Scanned slices retain the bytes
whose identity and metadata were validated. Pixel decoding consumes those
bytes, so replacing a path after scanning cannot substitute another image.

An existing DICOMDIR is authoritative. Invalid or missing references fail
instead of falling back to unrelated folder contents. Reference checks reject
absolute paths, traversal, and canonical paths outside the file-set root.
The current filesystem implementation assumes the file set remains unchanged
between reference validation and reading; it does not provide confinement
against concurrent filesystem replacement.
