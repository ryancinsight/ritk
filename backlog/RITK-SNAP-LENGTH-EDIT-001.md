<a id="RITK-SNAP-LENGTH-EDIT-001"></a>
## RITK-SNAP-LENGTH-EDIT-001 — Edit completed length endpoints — todo
- outcome: Pan or Length edits either endpoint of a completed image-space Length while preserving image calibration.
- acceptance: Native toolbar events drag each endpoint in anisotropic spacing and assert both `[row, column]` vertices, analytic `2A` then `3A` physical lengths, rendered preview, 2x calibration, source identity, and Escape/capture-loss/slice-return cancellation.
- scope: crates/ritk-snap/src/app/annotation_edit*, crates/ritk-snap/src/presentation/native_session/layout/overlay_measurements*
- next: Land endpoint geometry and native value-semantic coverage after the shared transaction prerequisite. Reference: RadiAnt 2021.2 page 75; [feature comparison](../docs/manual/features.md#reading-and-annotations)
- basis: 405c9963ef1fa738d18c61c99c5b9a0218f9480b
- needs: RITK-SNAP-ANNOTATION-EDIT-LIFECYCLE-001
- priority: feature; milestone: M-viewer-reading
