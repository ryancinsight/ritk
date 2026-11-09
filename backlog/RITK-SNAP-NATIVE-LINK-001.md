<a id="RITK-SNAP-NATIVE-LINK-001"></a>
## RITK-SNAP-NATIVE-LINK-001 — Link panes by patient position — todo
- outcome: Same-frame series scroll in sync, different-plane panes show cross-reference lines, the 3D cursor works in grids, and Home/End select the first/last image.
- acceptance: Native tests assert synchronized slice positions from known geometry and reference-line endpoints against an analytical plane intersection.
- scope: `crates/ritk-snap/src/presentation/native_session/{compare,panels,events}.rs`, `crates/ritk-snap/src/presentation/native_session/layout/crosshair.rs`
- next: Add a patient-position sync model over `ComparePanel`.
- basis: 30b8ba098
- status: todo
- needs: none
- milestone: M-viewer-compare
- priority: feature
