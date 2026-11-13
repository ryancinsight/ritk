<a id="RITK-SNAP-EFRAME-PORT-002"></a>
## RITK-SNAP-EFRAME-PORT-002 — Port morphology, geometry, CPR and pointwise filter ranges — todo
- outcome: The slider and drag ranges in `ui/filter_panel/controls_{morph,geom,cpr,pointwise}.rs` join `filter::parameter_range`. Egui code keeps its own copy until its deletion item; no egui line changes.
- acceptance: Value tests assert each added range against those files at the basis.
- scope: `crates/ritk-snap/src/filter/parameter_range*`
- next: Extend the table with the morphology radius and voting thresholds first.
- basis: 6d0f66501
- needs: RITK-SNAP-EFRAME-PORT-001
- milestone: M-viewer-reading
- priority: architecture
