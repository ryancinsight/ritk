<a id="RITK-SNAP-NATIVE-VR-001"></a>
## RITK-SNAP-NATIVE-VR-001 — Render volumes in the native shell — blocked
- outcome: A native 3D pane provides volume rendering with presets, rotate/zoom, snapshots, a scalpel, and STL export. STL, OBJ and PLY surface meshes load and draw over the 3D pane.
- acceptance: Golden renders of an analytical phantom pass; the exported STL's surface area matches the phantom within a derived bound.
- scope: `crates/ritk-snap/src/render/gpu_volume/`, `crates/ritk-snap/src/render/mip_vr.rs`, `crates/ritk-snap/src/presentation/native_session/`, `crates/ritk-snap/src/app/mesh_ops*`
- next: Route the existing renderer into a native pane.
- basis: 30b8ba098
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-SNAP-NATIVE-LOAD-001
- milestone: M-viewer-advanced
- priority: feature
