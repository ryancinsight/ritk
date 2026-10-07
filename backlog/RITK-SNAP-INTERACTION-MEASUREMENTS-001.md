<a id="RITK-SNAP-INTERACTION-MEASUREMENTS-001"></a>

## RITK-SNAP-INTERACTION-MEASUREMENTS-001 — Separate measurement tests — blocked
- outcome: give measurement tools one test module without changing behavior.
- acceptance: moved measurement tests preserve their existing value oracles and pass.
- scope: `crates/ritk-snap/src/tools/interaction/tests.rs`, `tools/interaction/tests/measurements.rs`
- next: move measurement cases after the region extraction lands.
- basis: 3fcdc3dd
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-SNAP-INTERACTION-REGIONS-001
- priority: tightening
