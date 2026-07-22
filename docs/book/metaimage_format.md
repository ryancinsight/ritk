# MetaImage Format Boundary

MetaImage is ritk's lightweight lossless boundary for header-driven volume interchange. The `ritk-io` facade exposes `read_metaimage`, `write_metaimage`, and backend-bound `MetaImageReader`/`MetaImageWriter` adapters over the native `ritk-metaimage` implementation, so `.mha` single-file payloads and `.mhd` plus raw-data pairs enter the toolkit through one consistent contract. The key responsibility of this chapter is not just pixel decode, but preservation of spacing, origin, direction, and validated `[depth, row, col]` shape at the `ritk-image::Image` boundary.

In Atlas terms, MetaImage is a clean bridge between file I/O and Coeus-backed image processing. Readers construct `Image<f32, B, 3>` values directly on the selected backend, while writers extract contiguous host data only at the final boundary. That means the same volume can be loaded once, passed through ritk filters, and dispatched later on Sequential or Moirai backends without changing the format-facing API. MetaImage is especially useful for debugging, golden-data fixtures, and simple round trips where DICOM's object model would be unnecessary overhead.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| Dedicated MetaImage round trip | Planned | Demonstrate `.mha` and `.mhd`/raw reads and writes through `MetaImageReader` and `MetaImageWriter`. |
| [DICOM to NIfTI Conversion](examples/dicom_to_nifti.md) | Available | Shows the same `ritk-image` boundary style used by format-to-format conversion workflows. |
