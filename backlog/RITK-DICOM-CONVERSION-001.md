<a id="RITK-DICOM-CONVERSION-001"></a>

## RITK-DICOM-CONVERSION-001 — Convert DICOM series through RITK — blocked
- outcome: Convert DICOM series through typed stored models and format adapters.
- acceptance: Supported pixels, geometry, calibration, units, acquisition axis, and inventoried metadata are preserved; unsupported source or target semantics are reported before destination mutation.
- scope: crates/ritk-io/src/format/dicom/, crates/ritk-image-io/, tests, and manual
- next: Implement the DICOM adapter after import and metadata preflight pass on real fixtures.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-DICOM-METADATA-INVENTORY-001, RITK-DICOM-STORED-IMPORT-001, RITK-DICOM-CALIBRATION-UNITS-001, RITK-DICOM-OBJECT-PIXEL-PREFLIGHT-001
- priority: architecture
