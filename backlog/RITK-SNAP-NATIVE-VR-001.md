<a id="RITK-SNAP-NATIVE-VR-001"></a>
## RITK-SNAP-NATIVE-VR-001 — Render volumes in the native shell — todo
- outcome: A native 3D pane provides volume rendering with presets, rotate/zoom, snapshots, a scalpel, and STL export. STL, OBJ and PLY surface meshes load and draw over the 3D pane.
- acceptance: Golden renders of an analytical phantom pass; the exported STL's surface area matches the phantom within a derived bound.
- scope: `crates/ritk-snap/src/render/gpu_volume/`, `crates/ritk-snap/src/render/mip_vr.rs`, `crates/ritk-snap/src/presentation/native_session/`, `crates/ritk-snap/src/app/mesh_ops*`
- next: Add the native VR seam (`render_vr_axial_rgba_into` + `OrthogonalWithVr` mode) with pixel-parity tests.
- basis: f7d05ac2f
- needs: none
- priority: feature; milestone: M-viewer-advanced
