# Target 1.0.0 · AsterSlice and RITK

The [release ladder](backlog.md#milestones) defines the milestones and their
exit checks. The current increment belongs to M-viewer-reading.

- [x] [patch] MGH 0.6.1 and AsterSlice 0.2.1 use FreeSurfer's voxel center and open both gzip suffixes. Three regressions fail on the baseline. All 629 MGH and viewer library tests pass on macOS. NiBabel independently confirms all 12 samples and the full affine through MGZ → NIfTI → MGH → compressed MGH → MGZ.
- [x] [minor] CLI 0.3.0 preserves all decompressed bytes when repacking a supported NIfTI document. The boundary tests cover a calibrated two-volume NIfTI-2 integer file and truncated input with an existing destination.
- [x] [patch] Verify Python VTK and Analyze physical coordinates against SimpleITK and reject unrepresentable Analyze stored headers. All 306 Analyze/VTK library tests pass.
- [ ] [minor] Implement the [stored MGH adapter](backlog/RITK-MGH-CONVERSION-001.md) and verify exact scalar types and metadata-loss rejection.
- [ ] [major] Execute the native Windows viewer and browser acceptance gates and complete the remaining [milestone items](backlog.md).
- [x] [major] Record the four 1.0 compatibility surfaces in [ADR 0059](docs/adr/0059-release-scope-1-0.md). Compatibility qualification remains pending.
