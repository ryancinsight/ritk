<a id="RITK-SNAP-OBLIQUE-ROUTING-001"></a>

## RITK-SNAP-OBLIQUE-ROUTING-001 — Route native oblique input — blocked
- outcome: route keyboard, pointer, wheel, and repaint events to the active oblique pane.
- acceptance: navigation changes only the oblique plane; linked cursor updates orthogonal panes; invalid events preserve visible state.
- scope: `crates/ritk-snap/src/presentation/native_session/routing.rs`, `events.rs`
- next: implement the bounded event state machine and user-visible controls.
- basis: 3fcdc3dd
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-SNAP-OBLIQUE-SESSION-WIRING-001, RITK-SNAP-OBLIQUE-APP-TESTS-001
- priority: feature
