<a id="RITK-SNAP-MEASUREMENT-UNITS-001"></a>
## RITK-SNAP-MEASUREMENT-UNITS-001 — Retain missing-spacing measurement units — todo
- outcome: A DICOM image without physical pixel spacing never displays an unqualified millimetre measurement.
- acceptance: A native synthetic DICOM fixture omits PixelSpacing, preserves its absence, asserts pixel units or explicit unavailable physical units, then verifies calibrated millimetres and reset; calibrated inputs retain anisotropic values.
- evidence: Original axial stacks with confirmed absent spacing retain pixel distances. Calibration still requires a millimetre reference. Pixel endpoint editing remains open.
- scope: crates/ritk-snap/src/loaded_volume.rs, dicom/loader/, app measurement/calibration paths, native measurement overlays/tests, and docs/manual/
- next: Admit a source-slice pixel reference for manual calibration using verified pixel aspect, and assert reset. Verify pixel endpoint edits and translation in the native session.
- needs: none
- priority: correctness; milestone: M-viewer-reading
- basis: eee4ce8c2f62f4a6a141301ae7253816b544a331
