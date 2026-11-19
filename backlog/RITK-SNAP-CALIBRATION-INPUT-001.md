<a id="RITK-SNAP-CALIBRATION-INPUT-001"></a>
## RITK-SNAP-CALIBRATION-INPUT-001 — Complete calibration units and properties — todo
- outcome: The calibration dialog accepts the reference manual's documented units and exposes calibration provenance.
- acceptance: Native dialog fixtures assert equivalent mm/inch/French-size inputs, invalid input rejection, reset, source-specific label properties, and per-image ownership.
- evidence: RadiAnt PDF 76–79 distinguishes calibration sources and accepts millimetres, inches and French sizes; the current dialog accepts known_mm only.
- scope: crates/ritk-snap/src/app/calibration.rs, presentation/native_session calibration controls, reusable Metis controls only where absent, and docs/manual/
- next: Verify authoritative unit definitions; reuse existing GUI controls and keep conversion/provenance in RITK.
- needs: RITK-SNAP-MEASUREMENT-UNITS-001
- priority: feature; milestone: M-viewer-reading
- basis: eee4ce8c2f62f4a6a141301ae7253816b544a331
