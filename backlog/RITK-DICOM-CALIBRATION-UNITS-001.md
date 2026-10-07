<a id="RITK-DICOM-CALIBRATION-UNITS-001"></a>

## RITK-DICOM-CALIBRATION-UNITS-001 — Preserve DICOM values and units — blocked
- outcome: Represent DICOM stored-to-physical transforms and units without changing stored samples.
- acceptance: Rescale slope/intercept, per-frame transforms, real-world value mappings, and coded units become typed values or scoped losses; rendering applies transforms once and conversions preserve or reject before output mutation.
- scope: crates/ritk-image-io/, crates/ritk-io/src/format/dicom/, calibration tests, and ADR 0055
- next: Derive the unit model from DICOM coded-value rules and IntensityCalibration.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-DICOM-METADATA-INVENTORY-001, RITK-DICOM-STORED-IMPORT-001
- priority: correctness
