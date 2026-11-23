<a id="RITK-SNAP-NATIVE-LINK-001"></a>
## RITK-SNAP-NATIVE-LINK-001 — Link panes by patient position — todo
- outcome: Same-frame series scroll in sync, different-plane panes show cross-reference lines, the 3D cursor works in grids. Series browsing is qualified under the [reading milestone](RITK-SNAP-RADIANT-UI-001.md).
- acceptance: Native tests assert synchronized slice positions from known geometry and reference-line endpoints against an analytical plane intersection.
- scope: `crates/ritk-snap/src/presentation/native_session/{compare,panels,events}.rs`, `crates/ritk-snap/src/presentation/native_session/layout/crosshair.rs`
- [x] [minor] Implement automatic slice matching. Require the same study, a shared verified Frame of Reference and qualified positions and normals. Compare the unoriented plane angle with the manual’s inclusive 5° limit. Select the nearest target plane to the source slice center in patient space. At an exact midpoint choose the lower displayed index. Preserve the target when the projected position lies outside its outer half-slice boundaries. Exercise unequal spacing, reversed order, boundary angles and incompatible or missing geometry.
- [x] [minor] Manual slice alignment. Preserve the selected anatomical anchors across series and test the manual’s inclusive 30° limit. Capture anchors without moving either series. Move by signed millimetres from each anchor, preserve the existing F5 auto/off control and discard anchors on replacement.
- [ ] [minor] Support linked pan/zoom and independently selectable window settings. Verify that disabled settings preserve each panel’s display.
- [x] [minor] Cross-reference lines. Verify patient-to-screen projection for intersecting planes, clipped endpoints and parallel planes. Draw only in inactive panes sharing the source frame.
- [ ] [minor] Project a selected patient point into all eligible comparison panes.
- next: Implement linked pan/zoom and optional window settings. At `247bc001b`, Windows and all eight hosted browser scenarios qualified the current comparison work. [Windows tests](https://github.com/ryancinsight/ritk/actions/runs/45123805890/job/137523992803) · [browser matrix](https://github.com/ryancinsight/ritk/actions/runs/45123808988). The [manual comparison](../docs/manual/features.md#reading-workstation-parity) records the reference behavior.
- basis: 30b8ba098
- needs: none
- priority: feature; milestone: M-viewer-compare
