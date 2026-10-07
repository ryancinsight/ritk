<a id="RITK-IMAGE-CONVERSION-ADAPTERS-001"></a>

## RITK-IMAGE-CONVERSION-ADAPTERS-001 — Implement production stored-volume adapters — todo
- outcome: Connect real NIfTI and NRRD codecs to stored-series conversion preflight.
- acceptance: Production adapters inspect real StoredSeries values and report every unsupported sample, geometry, calibration, axis, or metadata before output opens; codec tests use real adapters.
- scope: crates/ritk-image-io/, crates/ritk-io/src/format/{nifti,nrrd}/, and conversion tests
- next: Implement target-owned NIfTI and NRRD plans and prove rejected writes preserve existing outputs. Both prerequisites have landed: `RITK-IO-FORMAT-CAPABILITIES-001` made the shared dispatch the single capability authority, and `RITK-NIFTI-STORED-READ-001` completed the NIfTI stored read. What remains is the adapter half — `ritk-nifti` and `ritk-nrrd` each expose `ConversionTarget` + `ConversionAdapter` and route through `prepare_conversion`, per `docs/architecture.md` Theorem 20.1 and the convergence step in section 22. `ritk-nrrd` currently validates inside `NrrdDocument::new` / `write_to` and exposes no adapter; `ritk-nifti` has `NiftiDocument::from_stored_series` but no `ConversionTarget` impl, so the shared preflight entry point has no production implementation yet.
- basis: 687fa435b6f14a202ff005c4da0a378ae8de0edc
- needs: none
- priority: architecture
