<a id="RITK-SNAP-NATIVE-RT-001"></a>
## RITK-SNAP-NATIVE-RT-001 — Show RT structures and dose natively — todo
- outcome: RT-STRUCT contours, RT-DOSE overlays, dose-volume histograms and RT plans load and display in the native panes. Native users can export a label map as RT-STRUCT.
- acceptance: Native tests assert contour endpoints and dose overlay pixels against analytical phantoms and the DVH against a known dose grid.
- scope: `crates/ritk-snap/src/app/rt_overlay.rs`, `crates/ritk-snap/src/ui/rtdose_overlay.rs`, `crates/ritk-snap/src/ui/rtstruct_overlay.rs`, `crates/ritk-snap/src/presentation/native_session/layout/`, `crates/ritk-snap/src/app/rt_struct_export.rs`
- next: Draw projected contours through the measurement overlay's display list.
- basis: 30b8ba098
- status: todo
- needs: none
- milestone: M-viewer-advanced
- priority: feature
