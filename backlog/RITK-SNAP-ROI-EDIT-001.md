<a id="RITK-SNAP-ROI-EDIT-001"></a>
## RITK-SNAP-ROI-EDIT-001 — Move and resize image regions — todo
- outcome: Rectangular and elliptical regions expose handles, retain their source slice, and update area plus min/max/mean/standard deviation after editing.
- acceptance: Drag an ROI on known anisotropic pixel values and assert geometry, all statistics, selection precedence, cancellation, and save/restore.
- scope: crates/ritk-snap/src/app/annotation_edit*, crates/ritk-snap/src/app/pointer_ops/measurement.rs, crates/ritk-snap/src/presentation/native_session/layout/
- next: Reuse the annotation handle seam and the existing ROI statistic computation. Reference: RadiAnt 2021.2; [feature comparison](../docs/manual/features.md#reading-and-annotations)
- basis: eb9a81dabb62777be2103fa16b8a2dcbab50f1e3
- needs: RITK-SNAP-LENGTH-EDIT-001
- priority: feature; milestone: M-viewer-reading
