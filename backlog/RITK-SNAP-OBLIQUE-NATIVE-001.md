<a id="RITK-SNAP-OBLIQUE-NATIVE-001"></a>

## RITK-SNAP-OBLIQUE-NATIVE-001 — Native oblique MPR — blocked
- outcome: deliver a native four-plane oblique viewer with patient-space navigation and measurement.
- acceptance: all child items land; invalid geometry is rejected; the public phantom capture shows one complete, uncropped app window with visible menus or toolbar buttons and all four anatomical panes; native visual and value-semantic gates pass.
- scope: `crates/ritk-snap/src/{app,presentation,render,tools/interaction,session}/`, `crates/ritk-snap/src/main.rs`, crate README, ADRs, manual, and provenance.
- next: deliver ready child items in dependency order; preserve the public MRI-DIR phantom as the shareable visual fixture.
- risk: [major] [arch]; patient-space annotations and session format 3; no registry release is authorized.
- basis: 4e2e797c6af199d0c33e37148e784b2becc04981
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-SNAP-INTERACTION-REGIONS-001, RITK-SNAP-INTERACTION-MEASUREMENTS-001, RITK-SNAP-INTERACTION-STATE-001, RITK-SNAP-INTERACTION-WINDOW-LEVEL-001, RITK-SNAP-OBLIQUE-APP-ADAPTER-001, RITK-SNAP-OBLIQUE-APP-TESTS-001, RITK-SNAP-OBLIQUE-SESSION-MODULES-001, RITK-SNAP-OBLIQUE-SESSION-WIRING-001, RITK-SNAP-OBLIQUE-ROUTING-001, RITK-SNAP-PATIENT-MEASUREMENT-OVERLAY-001, RITK-SNAP-OBLIQUE-SESSION-TESTS-001, RITK-SNAP-OBLIQUE-INTERACTION-TESTS-001, RITK-SNAP-OBLIQUE-MANUAL-001
- priority: architecture
