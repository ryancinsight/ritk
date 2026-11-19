<a id="RITK-SNAP-ROI-NATIVE-001"></a>
## RITK-SNAP-ROI-NATIVE-001 — Test native ROI creation and labels — todo
- outcome: Native ROI menu and pointer routes have value-semantic coverage rather than a length-only test citation.
- acceptance: Drive rectangular and elliptical ROI creation on known anisotropic pixels; assert source slice, geometry, mean, standard deviation, min/max, physical area, rendered label values and pixels, cancellation, and adjacent-slice exclusion.
- scope: crates/ritk-snap/src/presentation/native_session/tests/, docs/manual/features.md
- next: Reuse the real native-session fixture and independent constant/graded-pixel references; distinguish computed min/max from their missing on-image presentation.
- links: [comparison](../docs/manual/features.md#reading-and-annotations); [milestone](../backlog.md#M-viewer-reading)
- needs: none
- priority: verification; milestone: M-viewer-reading
- basis: f6e3164af3ccb839e8e99bb2caf5fc3cb39b3f8d
