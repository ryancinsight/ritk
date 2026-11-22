# Target 1.0.0 · AsterSlice and RITK

The [release ladder](backlog.md#milestones) defines the milestones and their
exit checks. The current increment belongs to M-viewer-reading.

- [x] [patch] MGH 0.6.1 and AsterSlice 0.2.1 use FreeSurfer's voxel center and open both gzip suffixes. Three regressions fail on the baseline. All 629 MGH and viewer library tests pass on macOS. NiBabel independently confirms all 12 samples and the full affine through MGZ → NIfTI → MGH → compressed MGH → MGZ.
- [x] [minor] CLI 0.3.0 preserves all decompressed bytes when repacking a supported NIfTI document. The boundary tests cover a calibrated two-volume NIfTI-2 integer file and truncated input with an existing destination.
- [x] [patch] Verify Python VTK and Analyze physical coordinates against SimpleITK and reject unrepresentable Analyze stored headers. All 306 Analyze/VTK library tests pass.
- [x] [patch] Preserve NIfTI gzip member boundaries on read by joining the expanded data and validating every document trailer. Two regressions failed on the baseline and all 106 NIfTI tests pass.
- [x] [patch] Inventory MetaImage conversion fields and correct big-endian reads using the alternate header key. All 25 library tests pass, including 16 packaging/key combinations.
- [x] [minor] Transfer decoded scalar DICOM pixels into the viewer with their resolved grid and release encoded inputs after decoding. Across seven affected packages, 1,666 Rust tests pass. All 196 applicable minor-release checks pass against the 0.8.0 source on main.
- [ ] [minor] Verify the native Display menu half-turn and its inverse on Windows. A rectangular DICOM fixture checks every presented pixel.
- [x] [minor] MGH 0.7.0 and the CLI repack FreeSurfer documents with exact header, frame, and footer bytes. Multi-frame signed integers, gzip trailer validation and in-place replacement are covered.
- [ ] [minor] Implement the [stored MGH adapter](backlog/RITK-MGH-CONVERSION-001.md) and verify exact scalar types and metadata-loss rejection.
- [ ] [major] Execute the native Windows viewer and browser acceptance gates and complete the remaining [milestone items](backlog.md).
- [x] [major] Record the four 1.0 compatibility surfaces in [ADR 0059](docs/adr/0059-release-scope-1-0.md). Compatibility qualification remains pending.
