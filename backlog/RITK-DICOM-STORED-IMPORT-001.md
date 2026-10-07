<a id="RITK-DICOM-STORED-IMPORT-001"></a>

## RITK-DICOM-STORED-IMPORT-001 — Retain DICOM stored pixel values — blocked
- outcome: Import supported DICOM image series into stored samples without scaling or narrowing voxels.
- acceptance: The initial path accepts monochrome uncompressed instances with identity calibration, validates geometry and pixel layout, preserves source metadata inventory, and returns exact samples; unsupported encoding or calibration fails before a series escapes.
- scope: crates/ritk-io/src/format/dicom/reader/, crates/ritk-image-io/, tests, and manual
- next: Use the source-owned inventory first; keep rescale and units in the dependent calibration item.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-DICOM-METADATA-INVENTORY-001
- priority: correctness
