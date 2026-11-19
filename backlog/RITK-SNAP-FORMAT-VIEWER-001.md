<a id="RITK-SNAP-FORMAT-VIEWER-001"></a>
## RITK-SNAP-FORMAT-VIEWER-001 — Exercise encoded studies in the viewer — todo
- outcome: Supported raw, color, multiframe, RLE, JPEG lossy/lossless, JPEG 2000 and JPEG-LS inputs have native visible-load value tests.
- acceptance: Load a tiny independent fixture for each declared transfer syntax through the actual native session and assert exact or codec-derived pixel values and slice identity.
- scope: crates/ritk-io/src/format/dicom/reader/tests/, crates/ritk-snap/src/dicom/loader/tests/, crates/ritk-snap/src/presentation/native_session/tests/
- next: Derive the declared syntax set from source and add missing loader-to-frame assertions. Reference: RadiAnt 2021.2; [feature comparison](../docs/manual/features.md#reading-and-annotations)
- basis: eb9a81dabb62777be2103fa16b8a2dcbab50f1e3
- needs: none
- priority: verification; milestone: M-viewer-reading
