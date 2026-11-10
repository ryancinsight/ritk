<a id="RITK-SNAP-NATIVE-DRAG-FEEDBACK-001"></a>
## RITK-SNAP-NATIVE-DRAG-FEEDBACK-001 — Keep pan and zoom drags independent of event batching — todo
- outcome: A pan or zoom drag ends at the same offset and zoom whether its moves arrive in one host batch or several.
- acceptance: A native test drives three equal moves in one batch and in three separate batches and asserts equal `pan_offset` and `zoom` in both, and asserts each equals the value the total pointer displacement defines.
- scope: `crates/ritk-snap/src/app/pointer_ops.rs`, `crates/ritk-snap/src/presentation/native_session/layout/geometry.rs`, `crates/ritk-snap/src/presentation/native_session/tests/pointer_capture.rs`
- next: `on_drag` receives positions mapped through a viewport that already holds the drag's own pan and zoom, so pan_n = (d_n - pan_(n-1)) / t; with t = 1, three +20 moves in separate batches give 20, 20, 40. Measure the gesture in screen coordinates, or map through the viewport captured at press.
- basis: 13d8176bc
- status: todo
- needs: none
- milestone: M-viewer-reading
- priority: correctness
