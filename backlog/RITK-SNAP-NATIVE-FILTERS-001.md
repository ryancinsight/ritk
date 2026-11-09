<a id="RITK-SNAP-NATIVE-FILTERS-001"></a>
## RITK-SNAP-NATIVE-FILTERS-001 — Apply image filters natively — todo
- outcome: Sharpen, smooth, edge and emboss display filters are selectable per pane.
- acceptance: Tests assert each kernel's output against hand-computed pixels on a small phantom.
- scope: `crates/ritk-snap/src/render/`, `crates/ritk-snap/src/presentation/native_session/window_actions.rs`
- next: Add display-only convolution ahead of the window mapping.
- basis: 30b8ba098
- status: todo
- needs: none
- milestone: M-viewer-advanced
- priority: feature
