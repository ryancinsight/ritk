<a id="RITK-SNAP-DICOM-GRAPHICS-001"></a>
## RITK-SNAP-DICOM-GRAPHICS-001 — Display DICOM graphic overlay planes — todo
- outcome: A stored DICOM Overlay Data bitplane renders in its image coordinates with a visibility control.
- acceptance: Decode a 1-bit overlay fixture and assert exact pixels before and after image transforms; malformed dimensions and truncated data return typed errors.
- scope: crates/ritk-io/src/format/dicom/reader/, crates/ritk-snap/src/loaded_volume.rs, crates/ritk-snap/src/presentation/native_session/
- next: Read the Overlay Data standard contract and add a failing encoded-overlay fixture.
- basis: eb9a81dabb62777be2103fa16b8a2dcbab50f1e3
- needs: none
- milestone: M-viewer-reading
- priority: feature
- reference: RadiAnt 2021.2; [feature comparison](../docs/manual/features.md#reading-and-annotations)
