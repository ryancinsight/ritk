<a id="RITK-SNAP-LENGTH-TRANSLATE-001"></a>
## RITK-SNAP-LENGTH-TRANSLATE-001 — Translate completed lengths — todo
- outcome: Pan translates both endpoints of a completed image-space Length without changing its physical length or calibration.
- acceptance: Native events translate the analytic `3A` segment and assert both vertices, unchanged `3A`, 2x calibration, source identity, rendered candidate-index preview, topmost family selection, tool isolation, and Escape/capture-loss/slice-return cancellation.
- scope: crates/ritk-snap/src/app/annotation_edit/length.rs, crates/ritk-snap/src/presentation/native_session/tests/length_edit.rs, docs/manual
- next: Land whole-line movement after endpoint editing, then document completed Length editing and its remaining family limits.
- basis: 405c9963ef1fa738d18c61c99c5b9a0218f9480b
- needs: RITK-SNAP-LENGTH-EDIT-001
- priority: feature; milestone: M-viewer-reading
