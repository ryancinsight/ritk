# Target 1.0.0 · AsterSlice and RITK

The [release ladder](backlog.md#milestones) defines the milestones and their
exit checks. The current increment belongs to M-viewer-reading.

- [x] [patch] MGH 0.6.1 and AsterSlice 0.2.1 use FreeSurfer's voxel center and open both gzip suffixes. Three regressions fail on the baseline. All 629 MGH and viewer library tests pass on macOS. NiBabel independently confirms all 12 samples and the full affine through MGZ → NIfTI → MGH → compressed MGH → MGZ.
- [ ] [minor] Implement the [stored MGH adapter](backlog/RITK-MGH-CONVERSION-001.md) and verify exact scalar types and metadata-loss rejection.
- [ ] [major] Execute the native Windows viewer and browser acceptance gates and complete the remaining [milestone items](backlog.md).
