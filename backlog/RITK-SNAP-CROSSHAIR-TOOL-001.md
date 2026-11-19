<a id="RITK-SNAP-CROSSHAIR-TOOL-001"></a>
## RITK-SNAP-CROSSHAIR-TOOL-001 — Enable the selected crosshair tool — todo
- outcome: Choosing Crosshair and clicking places a visible linked cursor in three-plane MPR.
- acceptance: Drive the Tools menu and a known image click; assert the exact linked voxel, show_crosshair true, and six projected cursor segments. Pan leaves visibility unchanged.
- scope: crates/ritk-snap/src/app/action_adapter.rs, crates/ritk-snap/src/presentation/native_session/tests/
- next: Route crosshair selection or click through the existing cursor visibility transition. Reference: RadiAnt 2021.2; [feature comparison](../docs/manual/features.md#reading-and-annotations)
- basis: eb9a81dabb62777be2103fa16b8a2dcbab50f1e3
- needs: none
- milestone: M-viewer-reading
- priority: correctness
