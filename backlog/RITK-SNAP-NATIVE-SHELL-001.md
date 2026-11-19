<a id="RITK-SNAP-NATIVE-SHELL-001"></a>
## RITK-SNAP-NATIVE-SHELL-001 — Finish native shell integration — todo
- outcome: Full-screen mode, customizable keyboard shortcuts, a shortcut help screen, program/version information, and command-line integration work in the native shell.
- acceptance: Native tests assert a rebound shortcut dispatches its action that shortcut help and product/version information display their actual values, and that launch arguments select the requested study.
- scope: `crates/ritk-snap/src/presentation/native_session/window_controls/events.rs`, `crates/ritk-snap/src/launch.rs`
- next: Move the key table behind a binding map.
- basis: 30b8ba098
- needs: none
- priority: feature; milestone: M-viewer-advanced
