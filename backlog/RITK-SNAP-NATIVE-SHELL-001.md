<a id="RITK-SNAP-NATIVE-SHELL-001"></a>
## RITK-SNAP-NATIVE-SHELL-001 — Finish native shell integration — todo
- outcome: Full-screen mode, customizable keyboard shortcuts, a shortcut help screen, and command-line integration work in the native shell.
- acceptance: Native tests assert a rebound shortcut dispatches its action and that the launch arguments select the requested study.
- scope: `crates/ritk-snap/src/presentation/native_session/window_controls/events.rs`, `crates/ritk-snap/src/launch.rs`
- next: Move the key table behind a binding map.
- basis: 30b8ba098
- status: todo
- needs: none
- milestone: M-viewer-advanced
- priority: feature
