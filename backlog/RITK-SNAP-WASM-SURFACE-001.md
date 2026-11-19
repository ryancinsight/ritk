<a id="RITK-SNAP-WASM-SURFACE-001"></a>
## RITK-SNAP-WASM-SURFACE-001 — Compile only live viewer state on WASM — todo
- outcome: The default WASM library compiles warning-denied without retaining state used only by desktop viewer shells.
- acceptance: the locked default `wasm32-unknown-unknown` library check passes with warnings denied and no dead-code suppression; Windows native overlay toggle and orientation-label value tests pass; non-WASM `eframe-shell` orientation and physical-aspect value tests pass; public `ViewerSessionSnapshot::show_overlay` and public LPS/RAS APIs remain unchanged.
- scope: `crates/ritk-snap/src/app/{state.rs,state_default.rs}`, `crates/ritk-snap/src/ui/coordinate_system.rs` and `crates/ritk-snap/src/ui/coordinate_system/orientation.rs`, and affected package checks
- next: gate `SnapApp::show_overlay` and its default to Windows native or non-WASM eframe consumers, move the unchanged private orientation-label family into one host-gated leaf to Windows or `eframe-shell`, then run default WASM, native, and eframe checks.
- basis: b2fe302b3659ec1dc69bba01849fab7f0e453d1c
- needs: none
- priority: verification; milestone: M-viewer-reading
