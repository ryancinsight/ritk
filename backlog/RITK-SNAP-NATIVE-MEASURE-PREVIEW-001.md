<a id="RITK-SNAP-NATIVE-MEASURE-PREVIEW-001"></a>
## RITK-SNAP-NATIVE-MEASURE-PREVIEW-001 — Preview a measurement while it is drawn — todo
- outcome: A measurement in progress draws its rubber band and live value on the pane before the point that completes it.
- acceptance: A native test asserts the rubber-band pixels and value mid-drag, and that cancelling the tool leaves the pane as it was before the press.
- scope: `crates/ritk-snap/src/presentation/native_session/`, `crates/ritk-snap/src/app/pointer_ops.rs`
- next: `draw_gesture` in `crates/ritk-snap/src/presentation/native_session/layout/measurement.rs` already draws the rubber band; add the live value label from `ui::live_preview` and the mid-drag and cancel tests. Deleted shell source: `draw_in_progress` in `414da7491:crates/ritk-snap/src/ui/measurements/mod.rs`.
- basis: 52ef00f99
- status: todo
- needs: none
- milestone: M-viewer-reading
- priority: feature
