<a id="RITK-SNAP-LENGTH-EDIT-001"></a>
## RITK-SNAP-LENGTH-EDIT-001 — Move and resize completed lengths — todo
- outcome: Pan selects a completed length on the active image; endpoint and whole-line drags recompute its physical value and preserve calibration.
- acceptance: A native anisotropic fixture selects and drags both endpoints and the line, asserts millimetres and source location, and proves Escape/capture-loss restore the original.
- scope: crates/ritk-snap/src/app/annotation_edit*, crates/ritk-snap/src/presentation/native_session/layout/overlay_measurements*
- next: Generalize the existing selected-annotation interaction around reusable handle picking, then add length edits. Reference: RadiAnt 2021.2; [feature comparison](../docs/manual/features.md#reading-and-annotations)
- basis: eb9a81dabb62777be2103fa16b8a2dcbab50f1e3
- needs: none
- milestone: M-viewer-reading
- priority: feature
