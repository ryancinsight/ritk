<a id="RITK-SNAP-INTERACTION-WINDOW-LEVEL-001"></a>

## RITK-SNAP-INTERACTION-WINDOW-LEVEL-001 — Separate window-level tests — blocked
- outcome: give window-level interaction tests one module without changing behavior.
- acceptance: moved window-level tests retain their value assertions and pass.
- scope: `crates/ritk-snap/src/tools/interaction/tests.rs`, `tools/interaction/tests/window_level.rs`
- next: move the remaining window-level cases and leave no implementation in the test manifest.
- basis: 3fcdc3dd
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-SNAP-INTERACTION-STATE-001
- priority: tightening
