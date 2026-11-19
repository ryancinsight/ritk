<a id="RITK-SNAP-CALIBRATION-MEASURES-001"></a>
## RITK-SNAP-CALIBRATION-MEASURES-001 — Propagate image calibration to measurements — todo
- outcome: Every supported physical measurement on the calibrated image uses the same manual scale.
- acceptance: Native fixtures assert scale s for lengths/deviation/perimeters and s squared for areas; angles and intensities stay unchanged; other images and reset retain their original readings.
- evidence: Native overlay applies length_scale only to Annotation::Length; PatientLength, deviation, ROI and polygon labels use stored values. RadiAnt PDF 79 describes recalibrating image measurements.
- scope: crates/ritk-snap/src/app/calibration.rs, presentation/native_session/layout/overlay_measurements*, existing measurement tests, and docs/manual/
- next: Preserve raw annotations; apply the per-image scale through one measurement-display contract and test all affected families.
- needs: none
- priority: correctness; milestone: M-viewer-reading
- basis: eee4ce8c2f62f4a6a141301ae7253816b544a331
