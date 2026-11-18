<a id="RITK-SNAP-MARK-EDIT-001"></a>
## RITK-SNAP-MARK-EDIT-001 — Edit completed angles and drawn marks — todo
- outcome: Angle, Cobb, deviation, arrow, and pencil marks support selection, movement, resizing where applicable, and deletion on their source image.
- acceptance: Native tests edit each mark's geometric controls and assert recomputed angles/distances or exact paths, direction, cancellation, and ownership.
- scope: crates/ritk-snap/src/app/annotation_edit*, crates/ritk-snap/src/presentation/native_session/
- next: Extend the shared annotation selection/control model after length editing.
- basis: eb9a81dabb62777be2103fa16b8a2dcbab50f1e3
- needs: RITK-SNAP-LENGTH-EDIT-001
- milestone: M-viewer-reading
- priority: feature
- reference: RadiAnt 2021.2; [feature comparison](../docs/manual/features.md#reading-and-annotations)
