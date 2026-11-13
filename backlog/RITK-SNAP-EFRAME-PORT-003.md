<a id="RITK-SNAP-EFRAME-PORT-003"></a>
## RITK-SNAP-EFRAME-PORT-003 — Port filter labels and starting parameters — todo
- outcome: Every selectable filter's label and starting parameters in `ui/filter_panel/selector/selector_values*.rs` is an egui-free `filter::catalog`, with the value tests of `ui/filter_panel/tests_{integrity,smoothing}.rs`. Egui code keeps its own copy until its deletion item; no egui line changes.
- acceptance: The catalog tests pass on the default build and list every `FilterKind` the selector offers at the basis. Reference: `origin/rescue/ritk-root-refactor-ritk-snap-remove-eframe:crates/ritk-snap/src/filter/catalog/`.
- scope: `crates/ritk-snap/src/filter/catalog/`
- next: Port `selector_values.rs` and its tests. Each PR stays within 400 changed lines; split by module.
- basis: 6d0f66501
- needs: none
- milestone: M-viewer-reading
- priority: architecture
