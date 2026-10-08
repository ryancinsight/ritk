<a id="RITK-DICOM-METADATA-INVENTORY-001"></a>

## RITK-DICOM-METADATA-INVENTORY-001 — Account for DICOM metadata before discard — done
- outcome: Track every DICOM field that an import or conversion reads, retains, rejects, or omits.
- acceptance: Nested sequences, private creator scopes, unknown elements, malformed values, and failed conversions are retained opaquely or reported as scoped losses before parser data is dropped; conversion preflight consumes the source-owned inventory.
- scope: crates/ritk-dicom/, crates/ritk-io/src/format/dicom/, crates/ritk-image-io/, DICOM tests, and ADR 0054/0055
- next: Landed. ADR 0055 claims the contract. `DicomPreservationSet` now carries `losses: Vec<DicomRetentionLoss>` alongside interpreted nodes and opaquely retained elements, and `DicomRetentionReason` (`SequenceItemsUnavailable`, `ValueBytesUnavailable`, `NestingDepthExceeded`) names every case where retention is impossible. The six former silent-drop paths are closed: the reader/preservation loop and `parse_sequence_item` now attempt opaque byte retention before recording a loss, and the nesting bound (unchanged at levels 0..=8) records `NestingDepthExceeded` at the boundary element instead of returning an empty subtree. `inventory::dicom_metadata_losses` projects the source-owned inventory into `ritk_image_io::FormatMetadataLoss::UnknownSemantics` scoped to the caller's `ConversionLocation`, which is the single representation `prepare_conversion` consumes; per-slice losses stay on their slice so each keeps an exact `ConversionLocation::Frame`. Tests: boundary loss, in-bound case, opaque nested binary retention with private classification, loss-counting `is_empty`, and the projection's scoping and field text (six new tests, 487 passed in `ritk-io`). Residual: the call site that passes the projection to `prepare_conversion` lands with RITK-DICOM-STORED-IMPORT-001, which is the item that first produces a `StoredSeries` for a DICOM source.
- basis: 7905e953da4df4876163b37a2c857bf2fa27888a
- status: done
- needs: none
- priority: correctness
