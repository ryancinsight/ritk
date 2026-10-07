<a id="RITK-NIFTI-STORED-READ-001"></a>

## RITK-NIFTI-STORED-READ-001 — Read NIfTI into stored series — todo
- outcome: Decode NIfTI documents into exact stored samples and typed series metadata.
- acceptance: NIfTI-1/2 preserve supported samples, spatial forms, units, calibration, and ordered axes; unsupported semantics return scoped typed loss before a partial series escapes.
- scope: crates/ritk-nifti/, crates/ritk-image-io/, NIfTI tests, and the guide
- next: Implement document-to-StoredSeries decoding and test each scalar representation and malformed input.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: todo
- needs: none
- priority: correctness
