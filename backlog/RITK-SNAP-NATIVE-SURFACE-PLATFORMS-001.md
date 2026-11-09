<a id="RITK-SNAP-NATIVE-SURFACE-PLATFORMS-001"></a>
## RITK-SNAP-NATIVE-SURFACE-PLATFORMS-001 — Run the desktop viewer on Linux and macOS — todo
- outcome: `ritk-snap` opens a study in a native Métis window on Linux and macOS.
- acceptance: On each platform the viewer opens a synthetic study through the shell and a capture matches the Windows golden within the derived rasterizer tolerance, and `run_app_with_options` no longer returns the unsupported-target error there.
- scope: `crates/ritk-snap/src/launch.rs`, `crates/ritk-snap/src/lib.rs`, `crates/ritk-snap/src/presentation/native_session/`
- next: Métis has no surface provider off Windows, so add one upstream in `metis-platform`, then lift the Windows gates in `lib.rs` and `launch.rs`.
- basis: 52ef00f99
- status: todo
- needs: none
- milestone: M-viewer-advanced
- priority: feature
