<a id="RITK-SNAP-EFRAME-PORT-001"></a>
## RITK-SNAP-EFRAME-PORT-001 — Port smoothing and segmentation filter ranges — todo
- outcome: The range and edit step of every smoothing and segmentation filter parameter that `ui/filter_panel/controls.rs` holds in egui sliders is a typed, egui-free `filter::parameter_range` table with a clamping contract. Egui code keeps its own copy until its deletion item; no egui line changes.
- acceptance: Value tests assert each range's bounds and step against `controls.rs` at the basis and the clamp of below, above, inside and NaN inputs; they fail without the module.
- scope: `crates/ritk-snap/src/filter/`
- next: Add `ParameterRange<T>` and the `controls.rs` constants test-first.
- basis: 6d0f66501
- needs: none
- milestone: M-viewer-reading
- priority: architecture
