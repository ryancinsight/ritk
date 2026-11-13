<a id="RITK-SNAP-RADIANT-INPUT-001"></a>
## RITK-SNAP-RADIANT-INPUT-001 — Let readers reassign mouse buttons — todo
- outcome: A reader can assign any viewer tool to the middle, right, back or forward button, and the assignment persists between runs. Ctrl-left drag adjusts window and Shift-left drag pans, matching RadiAnt's documented 2D modifier gestures.
- acceptance: A native test assigns a tool to each button, asserts the resulting gesture, and asserts the assignment survives a restart. Native tests press with each modifier and assert window/level or pan values while the active tool stays unchanged.
- scope: `crates/ritk-snap/src/app/pointer_bindings.rs`, `crates/ritk-snap/src/presentation/native_session/window_controls/`, `crates/ritk-snap/src/presentation/{events,native,actions}.rs`, `crates/ritk-snap/src/app/pointer_bindings.rs`
- next: Add assignment actions over `PointerBindings`.
- basis: 30b8ba098
- needs: none
- milestone: M-viewer-reading
- priority: feature
