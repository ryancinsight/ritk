<a id="RITK-SNAP-NATIVE-SEGMENTATION-001"></a>
## RITK-SNAP-NATIVE-SEGMENTATION-001 — Draw and edit label maps natively — todo
- outcome: Native panes draw the label map through the view transform with hidden labels skipped and a visibility toggle, and label paint edits it with undo and redo on Ctrl+Z, Ctrl+Y and Ctrl+Shift+Z.
- acceptance: Native tests assert overlay pixels for a known label map under every display transform, that a hidden label draws nothing, and that undo then redo restores the edited voxels.
- scope: `crates/ritk-snap/src/presentation/native_session/`, `crates/ritk-snap/src/app/session_state.rs`, `crates/ritk-snap/src/app/action_adapter.rs`
- next: Rebuild the overlay from `draw_label_overlay` in `414da7491:crates/ritk-snap/src/app/rt_overlay.rs`, and bind the existing `undo_label_edit_shortcut` and `redo_label_edit_shortcut`.
- basis: 60878d06e
- status: todo
- needs: none
- milestone: M-viewer-advanced
- priority: feature
