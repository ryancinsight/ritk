<a id="RITK-SNAP-RADIANT-UI-001"></a>
## RITK-SNAP-RADIANT-UI-001 — Organize a multi-series DICOM viewer workspace — todo
- outcome: Present a RadiAnt-style diagnostic workspace with patient/study browser, series list, and organized image panes.
- acceptance: Multiple real DICOM series load into independently selectable viewports; menus, toolbar buttons, navigation, and pane controls work in end-to-end tests and appear in a full-window public-phantom capture; native and eframe orientation controls continue to share `ui::coordinate_system`.
- [patch] The web and Windows apps share the workstation. Chromium loads the 94-image MRI, compares CT and MR in two panels and passes the three-plane image checks.
- scope: crates/ritk-snap/src/{app,presentation,session,ui/coordinate_system.rs}, integration tests, manual, and image provenance
- next: Qualify the workstation in Firefox and Safari and run the Windows suite on the integrated revision.
- basis: b2fe302b3659ec1dc69bba01849fab7f0e453d1c
- needs: none
- priority: feature; milestone: M-viewer-reading
