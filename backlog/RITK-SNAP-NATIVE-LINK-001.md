<a id="RITK-SNAP-NATIVE-LINK-001"></a>
## RITK-SNAP-NATIVE-LINK-001 — Link panes by patient position — todo
- outcome: Same-frame series scroll in sync, different-plane panes show cross-reference lines, the 3D cursor works in grids. Series browsing is qualified under the [reading milestone](RITK-SNAP-RADIANT-UI-001.md).
- acceptance: Native tests assert synchronized slice positions from known geometry and reference-line endpoints against an analytical plane intersection.
- scope: `crates/ritk-snap/src/presentation/native_session/{compare,panels,events}.rs`, `crates/ritk-snap/src/presentation/native_session/layout/crosshair.rs`
- [x] [minor] Implement automatic slice matching. Require the same study, a shared verified Frame of Reference and qualified positions and normals. Compare the unoriented plane angle with the manual’s inclusive 5° limit. Select the nearest target plane to the source slice center in patient space. At an exact midpoint choose the lower displayed index. Preserve the target when the projected position lies outside its outer half-slice boundaries. Exercise unequal spacing, reversed order, boundary angles and incompatible or missing geometry.
- [ ] [minor] Manual alignment. Preserve the selected anatomical anchors across series and test the manual’s inclusive 30° limit. Expose automatic, manual and disabled modes, with separate pan/zoom and window settings.
- [ ] [minor] Cross-reference lines and grid cursor. Verify patient-to-screen projection for intersecting planes, clipped endpoints and parallel planes. Verify the same patient point in each eligible pane.
- next: Qualify Windows alongside the passing Chromium paired-stack journey, then implement reader-chosen anchors. Reference behaviors and sources are in the [manual comparison](../docs/manual/features.md#reading-workstation-parity).
- basis: 30b8ba098
- needs: none
- priority: feature; milestone: M-viewer-compare
