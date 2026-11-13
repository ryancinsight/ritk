<a id="RITK-SNAP-NATIVE-PROJECTION-001"></a>
## RITK-SNAP-NATIVE-PROJECTION-001 — Select projections from the native menu — todo
- outcome: MIP, MinIP and average projections with slab thickness are selectable at runtime instead of only at launch.
- acceptance: Native tests assert each projection's pixels against an analytical phantom at two thicknesses.
- scope: `crates/ritk-snap/src/presentation/native_session/{projection,window_actions}.rs`
- next: Add runtime layout and thickness actions.
- basis: 30b8ba098
- needs: none
- milestone: M-viewer-compare
- priority: feature
