<a id="RITK-SNAP-CALIBRATION-INPUT-001"></a>
## RITK-SNAP-CALIBRATION-INPUT-001 — Expose calibration sources and properties — todo
- outcome: The calibration controls expose the reference manual's calibration provenance and source-specific appearance.
- acceptance: Native fixtures assert segment/label context calibration, mouse apply/cancel, source-specific label properties, provenance, reset, and per-image ownership.
- evidence: RadiAnt PDF 76–79 distinguishes calibration sources; source-specific colors and properties remain absent.
- scope: crates/ritk-snap/src/app/calibration.rs, presentation/native_session calibration controls, reusable Metis controls only where absent, and docs/manual/
- next: Add segment/label context calibration actions; verify source tags and expose provenance and label properties through existing GUI controls.
- needs: RITK-SNAP-MEASUREMENT-UNITS-001
- priority: feature; milestone: M-viewer-reading
- basis: 3e48c8663fbacdf048918f541ff885aaa903dc9b
