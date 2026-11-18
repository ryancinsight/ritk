<a id="RITK-SNAP-US-REGION-001"></a>
## RITK-SNAP-US-REGION-001 — Measure calibrated ultrasound regions — todo
- outcome: Length and ROI measurements use the calibrated ultrasound region containing the sampled image coordinates.
- acceptance: Use known ultrasound-region tags with unequal axis units/scales, assert physical lengths and areas, and reject incompatible or absent calibration explicitly.
- scope: crates/ritk-io/src/format/dicom/, crates/ritk-snap/src/loaded_volume.rs, crates/ritk-snap/src/tools/
- next: Read the ultrasound region standard and preserve its calibration at DICOM ingress.
- basis: eb9a81dabb62777be2103fa16b8a2dcbab50f1e3
- needs: none
- milestone: M-viewer-reading
- priority: feature
- reference: RadiAnt 2021.2; [feature comparison](../docs/manual/features.md#reading-and-annotations)
