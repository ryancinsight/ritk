<a id="RITK-SNAP-INTERACTION-STATE-001"></a>

## RITK-SNAP-INTERACTION-STATE-001 — Separate interaction state tests — blocked
- outcome: give tool-state transitions one test module without changing behavior.
- acceptance: moved transition tests preserve valid and invalid state outcomes and pass.
- scope: `crates/ritk-snap/src/tools/interaction/tests.rs`, `tools/interaction/tests/state.rs`
- next: move the state cases after measurement tests.
- basis: 3fcdc3dd
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-SNAP-INTERACTION-MEASUREMENTS-001
- priority: tightening
