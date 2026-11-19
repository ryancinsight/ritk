<a id="RITK-SNAP-NATIVE-DISPLAY-001"></a>
## RITK-SNAP-NATIVE-DISPLAY-001 — Route display controls in the native shell — todo
- outcome: Add the explicit 180-degree display action to the native shell.
- acceptance: Native menu tests assert a direct 180-degree action and transformed corner pixels.
- scope: crates/ritk-snap/src/presentation/native_session/window_actions.rs, crates/ritk-snap/src/presentation/native_session/window_controls/, crates/ritk-snap/src/presentation/native_session/tests/display_controls.rs
- next: Add the remaining direct rotation action; existing preset and transform tests remain unchanged.
- basis: 746c615198743d9b946dac2014cf3a0776f20f14
- needs: none
- priority: feature; milestone: M-viewer-reading
