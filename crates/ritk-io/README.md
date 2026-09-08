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

The bounded DICOM reader scan and series-load entry points have a `*_with_budget`
form accepting the typed `DicomReadBudget`. Its parser component is the Atlas
`ritk_dicom::ParseBudget`; the other fields set independent retained-study and
decoded-workspace ceilings. Construct one with
`DicomReadBudget::try_new(parser, max_retained_bytes, max_decoded_bytes)` when a
host needs limits below the finite default. Structural validation runs before
dicom-rs object materialization, retained bytes are charged before each slice
is stored, and the loader checks the planned peak frame/resample/volume
workspace before allocation. Budgeted filesystem reads resolve a path once,
compare the opened handle with the resolved path metadata, and read from that
handle. Scanned slices retain those validated bytes for later decoding.

An existing DICOMDIR is authoritative. Its index is subject to the parser
budget, and invalid or missing references fail instead of falling back to
unrelated folder contents. Reference checks reject absolute paths, traversal,
and canonical paths outside the file-set root. The reader retains validated
member bytes, so replacing a path after scanning cannot substitute another
image during decode. The handle metadata check detects replacement between path
resolution and handle inspection. On Unix, the final open uses `O_NOFOLLOW`; on
Windows, it requests a reparse-point handle and rejects a final reparse point.
Parent-directory traversal through directory handles remains a platform
integration boundary.
