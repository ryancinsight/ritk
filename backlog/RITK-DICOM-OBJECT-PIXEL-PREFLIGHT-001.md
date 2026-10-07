<a id="RITK-DICOM-OBJECT-PIXEL-PREFLIGHT-001"></a>

## RITK-DICOM-OBJECT-PIXEL-PREFLIGHT-001 — Validate DICOM pixels before writing — in-progress
- outcome: Reject DICOM objects whose metadata cannot describe their encoded payload before opening the destination.
- acceptance: Validate rows, columns, samples, frames, BitsAllocated/BitsStored/HighBit/PixelRepresentation, pixel VR, checked length, and legal padding; absent NumberOfFrames means one, and rejection preserves an existing destination.
- scope: crates/ritk-io/src/format/dicom/writer_object.rs, writer modules, and DICOM writer tests
- next: Preflight landed in writer/pixel_preflight.rs and is wired into both image write paths (writer/series.rs write_series_flat, writer/metadata.rs write_dicom_series_with_metadata); every slice is validated before any serialized slice reaches write_series_files, so a rejection preserves an existing destination. writer_object.rs must stay a verbatim emitter: dicom_multiframe_rejects_declared_frame_count_mismatch deliberately writes a declared-frame-count mismatch to exercise the reader, so guarding that path would break negative fixtures. Re-derive the remaining slice from PRs #792/#793 only if a non-verbatim image-object writer is actually required.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: todo
- needs: none
- priority: correctness
