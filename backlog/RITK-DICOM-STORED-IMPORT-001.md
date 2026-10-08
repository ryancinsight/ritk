<a id="RITK-DICOM-STORED-IMPORT-001"></a>

## RITK-DICOM-STORED-IMPORT-001 — Retain DICOM stored pixel values — todo
- outcome: Import supported DICOM image series into stored samples without scaling or narrowing voxels.
- acceptance: The initial path accepts monochrome uncompressed instances with identity calibration, validates geometry and pixel layout, preserves source metadata inventory, and returns exact samples; unsupported encoding or calibration fails before a series escapes.
- scope: crates/ritk-io/src/format/dicom/reader/, crates/ritk-image-io/, tests, and manual
- next: Build the `StoredSeries` for a DICOM source and pass `inventory::dicom_metadata_losses` to `prepare_conversion` before opening any destination. Keep rescale and units in the dependent calibration item.
- basis: e0316a43a2f73ddfe5b525d82b7360071fcfa519
- status: todo
- needs: none
- priority: correctness
