<a id="RITK-IMAGE-CONVERSION-ADAPTERS-001"></a>

## RITK-IMAGE-CONVERSION-ADAPTERS-001 — Implement production stored-volume adapters — blocked
- outcome: Connect real NIfTI and NRRD codecs to stored-series conversion preflight.
- acceptance: Production adapters inspect real StoredSeries values and report every unsupported sample, geometry, calibration, axis, or metadata before output opens; codec tests use real adapters.
- scope: crates/ritk-image-io/, crates/ritk-io/src/format/{nifti,nrrd}/, and conversion tests
- next: Implement target-owned NIfTI and NRRD plans and prove rejected writes preserve existing outputs.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-IO-FORMAT-CAPABILITIES-001, RITK-NIFTI-STORED-SERIES-001, RITK-NIFTI-STORED-READ-001
- priority: architecture
