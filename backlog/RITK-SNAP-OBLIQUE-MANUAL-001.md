<a id="RITK-SNAP-OBLIQUE-MANUAL-001"></a>

## RITK-SNAP-OBLIQUE-MANUAL-001 — Demonstrate oblique MPR in the user manual — blocked
- outcome: document native launch and controls with a genuine public MRI-DIR phantom capture.
- acceptance: CLI, README, and manual show the 94-file public phantom in a complete app window with working menus or toolbar controls, pane controls, and all four anatomical panes; provenance identifies the exact capture revision.
- scope: `crates/ritk-snap/src/{launch.rs,main.rs}`, crate README, `docs/manual/`, provenance tests
- next: capture the completed viewer from the public phantom and validate the manual artifacts.
- basis: 4e2e797c6af199d0c33e37148e784b2becc04981
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-SNAP-OBLIQUE-INTERACTION-TESTS-001, RITK-SNAP-OBLIQUE-SESSION-TESTS-001
- priority: verification
