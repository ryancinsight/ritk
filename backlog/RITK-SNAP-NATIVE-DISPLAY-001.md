<a id="RITK-SNAP-NATIVE-DISPLAY-001"></a>
## RITK-SNAP-NATIVE-DISPLAY-001 — Route display controls in the native shell — todo
- outcome: Window presets with number keys, negative mode, color maps, and custom presets are reachable from native menus and keys. Rotate 90 CW/CCW, 180, flip horizontal/vertical, fit and 1:1 sizing are reachable from native menus and change the presented pixels. The window panel shows a log-scaled intensity histogram with the window and level marked. A colorbar labels the window range beside each pane, and flip, rotate and fit have assigned keys.
- acceptance: Native tests assert the window values each preset applies per modality, inverted output pixels, and the rendered color-map output. Native tests assert the transformed corner pixels and that image-point mapping stays exact under each transform.
- scope: `crates/ritk-snap/src/presentation/native_session/window_controls/**`, `crates/ritk-snap/src/presentation/native_session/window_actions.rs`, `crates/ritk-snap/src/ui/window_presets/`, `crates/ritk-snap/src/presentation/native_session/window_actions.rs`, `crates/ritk-snap/src/presentation/native_session/window_controls/layout/construction.rs`, `crates/ritk-snap/src/ui/histogram.rs`
- next: Add `WindowAction` variants over the existing preset library. Deleted shell source: `414da7491:crates/ritk-snap/src/ui/colorbar.rs`, `414da7491:crates/ritk-snap/src/ui/preset_panel.rs`, `414da7491:crates/ritk-snap/src/app/shortcuts.rs`, `414da7491:crates/ritk-snap/src/app/panels.rs`.
- basis: 30b8ba098
- status: todo
- needs: none
- milestone: M-viewer-reading
- priority: feature
