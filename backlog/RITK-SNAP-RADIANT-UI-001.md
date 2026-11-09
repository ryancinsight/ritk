<a id="RITK-SNAP-RADIANT-UI-001"></a>

## RITK-SNAP-RADIANT-UI-001 — Organize a multi-series DICOM viewer workspace — todo
- outcome: Present a RadiAnt-style diagnostic workspace with patient/study browser, series list, and organized image panes.
- acceptance: Multiple real DICOM series load into independently selectable viewports; menus, toolbar buttons, navigation, and pane controls work in end-to-end tests and appear in a full-window public-phantom capture; `ui::coordinate_system` is wired to orientation controls or removed with its exports.
- scope: crates/ritk-snap/src/{app,presentation,session,ui/coordinate_system.rs}, integration tests, manual, and image provenance
- next: Wire or remove `ui::coordinate_system`, then capture the full shell with its menus, browser, and panes.
- basis: 30b8ba098
- status: todo
- needs: none
- milestone: M-viewer-reading
- priority: feature
