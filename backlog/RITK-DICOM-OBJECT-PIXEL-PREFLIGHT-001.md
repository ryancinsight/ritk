<a id="RITK-DICOM-OBJECT-PIXEL-PREFLIGHT-001"></a>

## RITK-DICOM-OBJECT-PIXEL-PREFLIGHT-001 — Validate DICOM pixels before writing — todo
- outcome: Reject DICOM objects whose metadata cannot describe their encoded payload before opening the destination.
- acceptance: Validate rows, columns, samples, frames, BitsAllocated/BitsStored/HighBit/PixelRepresentation, pixel VR, checked length, and legal padding; absent NumberOfFrames means one, and rejection preserves an existing destination.
- scope: crates/ritk-io/src/format/dicom/writer_object.rs, writer modules, and DICOM writer tests
- next: Re-derive a bounded slice from PRs #792/#793; do not promise transactional I/O.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: todo
- needs: none
- priority: correctness
