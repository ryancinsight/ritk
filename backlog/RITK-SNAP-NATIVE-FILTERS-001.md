<a id="RITK-SNAP-NATIVE-FILTERS-001"></a>
## RITK-SNAP-NATIVE-FILTERS-001 — Apply image filters natively — todo
- outcome: Sharpen, smooth, edge and emboss display filters are selectable per pane. The 59-filter volume catalog in `filter/catalog` runs from a native panel with its starting parameters and validated ranges.
- acceptance: Tests assert each kernel's output against hand-computed pixels on a small phantom.
- scope: `crates/ritk-snap/src/render/`, `crates/ritk-snap/src/presentation/native_session/window_actions.rs`, `crates/ritk-snap/src/filter/catalog/`, `crates/ritk-snap/src/app/filter*`
- next: Add display-only convolution ahead of the window mapping. Edit ranges come from `filter::parameter_range` and starting parameters from `filter::catalog`.
- basis: 30b8ba098
- needs: RITK-SNAP-EFRAME-PORT-001, RITK-SNAP-EFRAME-PORT-002, RITK-SNAP-EFRAME-PORT-003
- milestone: M-viewer-advanced
- priority: feature
