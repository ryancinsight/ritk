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
