<a id="RITK-SNAP-NATIVE-RELEASE-ACTIVATION-001"></a>
## RITK-SNAP-NATIVE-RELEASE-ACTIVATION-001 — Stop pointer releases activating a panel in keyboard routing — todo
- outcome: A pointer release never changes the panel that keyboard events after it reach, so keyboard routing agrees with the panel activation in `apply_events`.
- acceptance: A native test presses in panel 0, releases over panel 1 in a grid, sends a key in the same batch, and asserts the key acts on panel 0.
- scope: `crates/ritk-snap/src/presentation/native_session/events/panel_state.rs`, `crates/ritk-snap/src/presentation/native_session/events.rs`, `crates/ritk-snap/src/presentation/native_session/tests/`
- next: `retain_event` activates on every routed pointer event through `is_pointer_event`, while `apply_events` skips releases through `is_release`. Use one predicate for both.
- basis: 13d8176bc
- status: todo
- needs: none
- milestone: M-viewer-reading
- priority: correctness
