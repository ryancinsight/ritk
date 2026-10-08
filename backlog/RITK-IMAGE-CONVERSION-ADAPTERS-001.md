<a id="RITK-IMAGE-CONVERSION-ADAPTERS-001"></a>

## RITK-IMAGE-CONVERSION-ADAPTERS-001 — Implement production stored-volume adapters — todo
- outcome: Connect real NIfTI and NRRD codecs to stored-series conversion preflight.
- acceptance: Production adapters inspect real StoredSeries values and report every unsupported sample, geometry, calibration, axis, or metadata before output opens; codec tests use real adapters.
- scope: crates/ritk-image-io/, crates/ritk-io/src/format/{nifti,nrrd}/, and conversion tests
- next: NRRD half remains. The NIfTI half has landed in `c47b609b2` (`feat(ritk-nifti): Build stored series`): `crates/ritk-nifti/src/stored_series.rs` implements `ConversionTarget` + `ConversionAdapter` (`NiftiStoredSeriesTarget` / `NiftiStoredSeriesPlan`, 15 typed rejections each carrying a `ConversionRejection::location`), and `NiftiDocument::from_stored_series` routes through `prepare_conversion` before any output exists; 13 tests. `ritk-nrrd` still validates inside `NrrdDocument::new` / `write_to` and exposes no adapter, so the shared preflight entry point has no NRRD production implementation. Next: implement `NrrdStoredSeriesTarget` the same way, route `ritk-io`'s NRRD write path through it, and prove a rejected write preserves an existing destination.
- basis: e0316a43a2f73ddfe5b525d82b7360071fcfa519
- needs: none
- priority: architecture
