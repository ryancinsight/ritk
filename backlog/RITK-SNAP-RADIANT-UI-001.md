<a id="RITK-SNAP-RADIANT-UI-001"></a>

## RITK-SNAP-RADIANT-UI-001 — Organize a multi-series DICOM viewer workspace — blocked
- outcome: Present a RadiAnt-style diagnostic workspace with patient/study browser, series list, and organized image panes.
- acceptance: Multiple real DICOM series load into independently selectable viewports; menus, toolbar buttons, navigation, and pane controls work in end-to-end tests and appear in a full-window public-phantom capture; `ui::coordinate_system` is wired to orientation controls or removed with its exports.
- scope: crates/ritk-snap/src/{app,presentation,session,ui/coordinate_system.rs}, integration tests, manual, and image provenance
- next: After multi-series behavior lands, capture the complete application shell with its menus, buttons, browser, and image panes.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-SNAP-MULTI-SERIES-001
- priority: feature
