<a id="RITK-SNAP-RADIANT-UI-001"></a>
## RITK-SNAP-RADIANT-UI-001 — Organize a multi-series DICOM viewer workspace — todo
- outcome: Present a RadiAnt-style diagnostic workspace with patient/study browser, series list, and organized image panes.
- acceptance: Multiple real DICOM series load into independently selectable viewports; menus, toolbar buttons, navigation, and pane controls work in end-to-end tests and appear in a full-window public-phantom capture; native and eframe orientation controls continue to share `ui::coordinate_system`.
- [patch] The web and Windows apps share the workstation. Chromium loads the 94-image MRI, compares CT and MR in two panels and passes the three-plane image checks.
- scope: crates/ritk-snap/src/{app,presentation,session,ui/coordinate_system.rs}, integration tests, manual, and image provenance
- [x] [patch] Separate series selection from image selection at both catalog boundaries. Preserve single-series Home/End and the inactive pane. The packaged CT/MR journey exercises both cases.
- [ ] [patch] Requalify the workstation changes with the current eight-host matrix, including the native Safari file chooser and the hosted WebGPU adapter. [Main verification](../gap_audit.md#saved-study-browser-qualification-patch)
- [x] [patch] Correct [display-coordinate navigation](RITK-SNAP-NAVIGATION-UNITS-001.md) and verify equivalent drags at different fits and rotations.
- [x] [minor] Implement [automatic patient-position linking](RITK-SNAP-NATIVE-LINK-001.md) and verify unequal-spacing and incompatible-frame cases in Chromium. The [manual comparison](../docs/manual/features.md#reading-workstation-parity) records the reference for each workflow.
- basis: b2fe302b3659ec1dc69bba01849fab7f0e453d1c
- needs: none
- priority: feature; milestone: M-viewer-reading
