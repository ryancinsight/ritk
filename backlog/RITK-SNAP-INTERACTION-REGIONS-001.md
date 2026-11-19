<a id="RITK-SNAP-INTERACTION-REGIONS-001"></a>
## RITK-SNAP-INTERACTION-REGIONS-001 — Separate interaction test families — todo
- milestone: M-viewer-compare
- outcome: separate region, measurement, state-transition, and window-level tests by operation family.
- acceptance: compare complete nextest test-name inventories before/after; retain every region/measurement value assertion, valid/invalid transition outcome, and window-level assertion; all moved tests pass.
- scope: `crates/ritk-snap/src/tools/interaction/tests.rs`, `crates/ritk-snap/src/tools/interaction/tests/`
- next: extract regions, measurements, state, then window-level into their concern modules; leave the parent as a module manifest without changing behavior.
- basis: c4c271b5b253b0ec92880747dbc67933b96ef6ca
- needs: none
- priority: tightening
