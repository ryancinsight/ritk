<a id="RITK-SNAP-ANNOTATION-EDIT-LIFECYCLE-001"></a>
## RITK-SNAP-ANNOTATION-EDIT-LIFECYCLE-001 — Share annotation edit transactions — todo
- outcome: Polygon edits use kind-neutral selection, pending transaction, lifecycle, and DICOM handle presentation without changing behavior.
- acceptance: Native polygon selection, vertex drag, edge insertion, preview, cancellation, deletion, and markers retain their values; a failed direct admission preserves every field of an established transaction.
- scope: crates/ritk-snap/src/app/annotation_edit.rs, crates/ritk-snap/src/app/pointer_bindings.rs, crates/ritk-snap/src/presentation/native_session/layout/overlay_measurements*
- next: Land the bounded mechanical lifecycle/selection rename, indexed admission, and marker extraction required by completed Length editing.
- basis: 405c9963ef1fa738d18c61c99c5b9a0218f9480b
- needs: none
- priority: architecture; milestone: M-viewer-reading
