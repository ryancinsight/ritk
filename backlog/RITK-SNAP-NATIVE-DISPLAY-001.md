<a id="RITK-SNAP-NATIVE-DISPLAY-001"></a>
## RITK-SNAP-NATIVE-DISPLAY-001 — Route display controls in the native shell — todo
- outcome: Finish the explicit 180-degree display action and correct pixel-scale sizing; retain delivered presets, colormaps, histogram, inversion, and orientation controls.
- acceptance: Native menu tests assert a direct 180-degree action and transformed corner pixels; pixel-scale correction meets RITK-SNAP-ZOOM-SCALE-001.
- scope: crates/ritk-snap/src/presentation/native_session/window_actions.rs, crates/ritk-snap/src/presentation/native_session/window_controls/, crates/ritk-snap/src/presentation/native_session/tests/display_controls.rs
- next: Add the remaining direct rotation action and reuse the corrected zoom policy; existing preset and transform tests remain unchanged.
- basis: 30ade461c3cbf50a286111a44213e195c2fe5d95
- needs: RITK-SNAP-ZOOM-SCALE-001
- milestone: M-viewer-reading
- priority: feature
