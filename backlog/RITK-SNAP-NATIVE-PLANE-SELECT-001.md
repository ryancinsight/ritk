<a id="RITK-SNAP-NATIVE-PLANE-SELECT-001"></a>
## RITK-SNAP-NATIVE-PLANE-SELECT-001 — Choose each pane's plane natively — todo
- outcome: A native menu sets each pane's anatomical plane, and compare panes offer axial, coronal and sagittal pairing presets.
- acceptance: Native tests select each plane for a pane and assert its presented axis and frame size, and assert a compare preset sets both panes to the chosen plane.
- scope: `crates/ritk-snap/src/presentation/native_session/window_controls.rs`, `crates/ritk-snap/src/presentation/native_session/window_actions.rs`, `crates/ritk-snap/src/app/state.rs`
- next: Add `WindowAction` variants that drive `dual_axes` and `compare_axes`. Deleted shell source: Image > Axis in `414da7491:crates/ritk-snap/src/app/menu.rs`, and the Axes pickers and presets in `414da7491:crates/ritk-snap/src/app/toolbar.rs`.
- basis: 52ef00f99
- status: todo
- needs: none
- milestone: M-viewer-compare
- priority: feature
