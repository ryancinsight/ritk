<a id="RITK-SNAP-ROI-LABEL-STATISTICS-001"></a>
## RITK-SNAP-ROI-LABEL-STATISTICS-001 — Show complete ROI label statistics — todo
- outcome: On-image rectangular and elliptical ROI labels show min/max and contributing-pixel count alongside mean, standard deviation, and physical area.
- acceptance: Native labels for a radius-two ellipse over a 5-by-5 row-major field 1 through 25 show exactly 13 pixels, min 3, max 23; rectangle cases assert their independently enumerated samples. Mean, standard deviation, area, calibration, and slice identity remain correct.
- scope: crates/ritk-snap/src/tools/interaction/, crates/ritk-snap/src/presentation/native_session/layout/, native ROI tests
- next: Carry the sample count from the existing RITK statistics operation; reuse Métis text measurement and placement. Reference: RadiAnt 2021.2 p.82.
- links: [comparison](../docs/manual/features.md#reading-and-annotations); [milestone](../backlog.md#M-viewer-reading)
- needs: none
- priority: feature; milestone: M-viewer-reading
- basis: f6e3164af3ccb839e8e99bb2caf5fc3cb39b3f8d
