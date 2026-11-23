<a id="RITK-SNAP-RADIANT-UI-001"></a>
## RITK-SNAP-RADIANT-UI-001 — Organize a multi-series DICOM viewer workspace — todo
- outcome: Present a RadiAnt-style diagnostic workspace with patient/study browser, series list, and organized image panes.
- acceptance: Multiple real DICOM series load into independently selectable viewports; menus, toolbar buttons, navigation, and pane controls work in end-to-end tests and appear in a full-window public-phantom capture; native and eframe orientation controls continue to share `ui::coordinate_system`.
- [patch] The web and Windows apps share the workstation. Chromium loads the 94-image MRI, compares CT and MR in two panels and passes the three-plane image checks.
- scope: crates/ritk-snap/src/{app,presentation,session,ui/coordinate_system.rs}, integration tests, manual, and image provenance
- [x] [patch] Separate series selection from image selection at both catalog boundaries. Preserve single-series Home/End and the inactive pane. The packaged CT/MR journey exercises both cases.
- [ ] [patch] Complete the two remaining hosted browser checks. Six scenarios and the Windows workspace tests passed at `2b4f6ccaf`. Safari stopped while opening the selected MRI. The [native Safari qualification](https://github.com/ryancinsight/ritk/pull/1110) covers the real picker separately. Hosted Chromium WebGPU stopped before initialization. Local Chromium WebGPU passed at both display densities. [Windows tests](https://github.com/ryancinsight/ritk/actions/runs/45105768737/job/137467394703) · [browser matrix](https://github.com/ryancinsight/ritk/actions/runs/45105771232)
- [x] [patch] Correct [display-coordinate navigation](RITK-SNAP-NAVIGATION-UNITS-001.md) and verify equivalent drags at different fits and rotations.
- [x] [minor] Implement [automatic patient-position linking](RITK-SNAP-NATIVE-LINK-001.md) and verify unequal-spacing and incompatible-frame cases in Chromium. The [manual comparison](../docs/manual/features.md#reading-workstation-parity) records the reference for each workflow.
- basis: b2fe302b3659ec1dc69bba01849fab7f0e453d1c
- needs: none
- priority: feature; milestone: M-viewer-reading
