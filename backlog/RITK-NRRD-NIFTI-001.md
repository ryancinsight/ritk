<a id="RITK-NRRD-NIFTI-001"></a>

## RITK-NRRD-NIFTI-001 — Convert NRRD and NIfTI volumes — blocked
- outcome: Convert representable NRRD and NIfTI volumes without changing sample bits or physical meaning.
- acceptance: Both directions preserve exact samples, LPS millimeter geometry, calibration, and axis semantics; unknown fields or unsupported units yield typed loss before output changes.
- scope: crates/ritk-io/, crates/ritk-nrrd/, crates/ritk-nifti/, pair tests, and manual
- next: Use production adapters for both directions with one shared preflight path.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-IMAGE-CONVERSION-ADAPTERS-001, RITK-NIFTI-STORED-SERIES-001, RITK-NIFTI-STORED-READ-001
- priority: correctness
