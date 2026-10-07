<a id="RITK-DICOM-METADATA-INVENTORY-001"></a>

## RITK-DICOM-METADATA-INVENTORY-001 — Account for DICOM metadata before discard — todo
- outcome: Track every DICOM field that an import or conversion reads, retains, rejects, or omits.
- acceptance: Nested sequences, private creator scopes, unknown elements, malformed values, and failed conversions are retained opaquely or reported as scoped losses before parser data is dropped; conversion preflight consumes the source-owned inventory.
- scope: crates/ritk-dicom/, crates/ritk-io/src/format/dicom/, crates/ritk-image-io/, DICOM tests, and ADR 0054/0055
- next: Claim ADR 0055, map every parse-and-skip path, then test malformed and private elements.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: todo
- needs: none
- priority: correctness
