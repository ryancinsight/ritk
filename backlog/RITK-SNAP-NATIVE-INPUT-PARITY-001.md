<a id="RITK-SNAP-NATIVE-INPUT-PARITY-001"></a>
## RITK-SNAP-NATIVE-INPUT-PARITY-001 — Finish native input parity — todo
- outcome: Ctrl-left/Shift-left gestures work in the Windows native viewer, and a reader's pointer-button assignments reload on the next launch.
- acceptance: A native end-to-end test presses with Control and Shift held and asserts window/level and pan change with the tool unchanged; a second test writes assignments through the settings store, restarts, and asserts restored bindings.
- scope: `moirai-pal`/`metis-platform` pointer events, `crates/ritk-snap/src/presentation/native.rs`, `native_session/startup.rs`, host settings file
- next: Add `ModifierState` to the provider press events and translate it in `native.rs`; define the settings file schema and restore `pointer_bindings` at startup.
- basis: bf7d80cd9
- needs: none
- priority: feature; milestone: M-viewer-reading
