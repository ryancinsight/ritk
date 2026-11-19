<a id="RITK-SNAP-MEASUREMENT-UNITS-001"></a>
## RITK-SNAP-MEASUREMENT-UNITS-001 — Retain missing-spacing measurement units — todo
- outcome: A DICOM image without physical pixel spacing never displays an unqualified millimetre measurement.
- acceptance: A native synthetic DICOM fixture omits PixelSpacing, preserves its absence, asserts pixel units or explicit unavailable physical units, then verifies calibrated millimetres and reset; calibrated inputs retain anisotropic values.
- evidence: reader/scan/finalize.rs uses unit spacing when absent, while LoadedVolume.metadata retains optional slice pixel_spacing; native length labels always append mm.
- scope: crates/ritk-snap/src/loaded_volume.rs, dicom/loader/, app measurement/calibration paths, native measurement overlays/tests, and docs/manual/
- next: Use retained per-slice metadata at measurement admission; keep display geometry separate from evidence of physical calibration.
- needs: none
- priority: correctness; milestone: M-viewer-reading
- basis: eee4ce8c2f62f4a6a141301ae7253816b544a331
