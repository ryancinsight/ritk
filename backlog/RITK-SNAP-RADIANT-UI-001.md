<a id="RITK-SNAP-RADIANT-UI-001"></a>
## RITK-SNAP-RADIANT-UI-001 — Organize a multi-series DICOM viewer workspace — todo
- outcome: Present a RadiAnt-style diagnostic workspace with patient/study browser, series list, and organized image panes.
- acceptance: Multiple real DICOM series load into independently selectable viewports; menus, toolbar buttons, navigation, and pane controls work in end-to-end tests and appear in a full-window public-phantom capture; native and eframe orientation controls continue to share `ui::coordinate_system`.
- scope: crates/ritk-snap/src/{app,presentation,session,ui/coordinate_system.rs}, integration tests, manual, and image provenance
- next: Exercise multi-series selection and pane controls end to end, then capture the full shell with its menus, browser, and panes.
- basis: b2fe302b3659ec1dc69bba01849fab7f0e453d1c
- needs: none
- milestone: M-viewer-reading
- priority: feature
