<a id="RITK-SNAP-NATIVE-OVERLAY-TEXT-001"></a>
## RITK-SNAP-NATIVE-OVERLAY-TEXT-001 — Draw clinical image overlays natively — todo
- outcome: Comparison-grid panes show the same clinical corners, orientation, ruler, cursor values, and privacy toggles as orthogonal panes.
- acceptance: For known direction cosines and spacing, comparison-grid tests assert letters, ruler length, cursor value, and absence of patient text when anonymized.
- scope: crates/ritk-snap/src/presentation/native_session/layout/compare/, crates/ritk-snap/src/presentation/native_session/tests/
- next: Reuse the existing orthogonal clinical-overlay composition for comparison panes; preserve shared text placement in Métis.
- basis: 30ade461c3cbf50a286111a44213e195c2fe5d95
- needs: none
- priority: feature; milestone: M-viewer-reading
