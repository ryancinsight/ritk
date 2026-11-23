<a id="RITK-SNAP-NATIVE-LINK-001"></a>
# Series comparison checklist
- target: AsterSlice 0.4.0 [minor]
- outcome: Same-frame series scroll in sync, different-plane panes show cross-reference lines, the 3D cursor works in grids. Series browsing is qualified under the [reading milestone](../../backlog/RITK-SNAP-RADIANT-UI-001.md).
- acceptance: Native and browser tests check the same screen displacement, relative magnification, window center and width, nearest patient plane and cursor pixels.
- scope: `crates/ritk-snap/src/presentation/native_session/{display_link,grid_cursor,synchronization}.rs`, `crates/ritk-snap/src/geometry/patient_stack/`
- [x] [minor] Implement automatic slice matching. Require the same study, a shared verified Frame of Reference and qualified positions and normals. Compare the unoriented plane angle with the manual’s inclusive 5° limit. Select the nearest target plane to the source slice center in patient space. At an exact midpoint choose the lower displayed index. Preserve the target when the projected position lies outside its outer half-slice boundaries. Exercise unequal spacing, reversed order, boundary angles and incompatible or missing geometry.
- [x] [minor] Manual slice alignment. Preserve the selected anatomical anchors across series and test the manual’s inclusive 30° limit. Capture anchors without moving either series. Move by signed millimetres from each anchor, preserve the existing F5 auto/off control and discard anchors on replacement.
- [x] [minor] Support linked pan/zoom and independently selectable window settings. Windows and browser input tests cover screen displacement, zoom ratio, independent window presets and link toggles.
- [x] [patch] Exercise a rotated 1:1 receiver outside the drag range. The browser scenario loads a 2 × 2 isotropic source and a 2 × 8 target with 2 mm row and 1 mm column spacing into panes of unequal width. Check the target’s screen extent through zoom-out, zoom-in and reversal, then apply a linked window preset while pan is independent.
- [x] [minor] Cross-reference lines. Verify patient-to-screen projection for intersecting planes, clipped endpoints and parallel planes. Draw only in inactive panes sharing the source frame.
- [x] [minor] Project a held patient cursor into same-frame images, navigate to the nearest plane and restore reference lines on release. The raster and WebGPU checks sample the rendered crosshair. Safari holds the cursor through its native input driver.

## Verification

The integrated implementation at `895f48fee` passed both new Windows interaction tests and all nine browser scenarios, including Intel and Apple-silicon Safari. The regression also starts the panes at different magnifications and checks their zoom ratio after relinking. The patient-geometry suite covers singleton images, rounded direction cosines, adjacent representable plane boundaries, out-of-field points and unrelated frames. At `db1880f4e`, switching from orthogonal views clears the persistent voxel crosshair. Resizing an existing comparison grid preserves the assigned patient cursor. [Windows results](https://github.com/ryancinsight/ritk/actions/runs/45214165566/job/137831240444) · [Browser matrix](https://github.com/ryancinsight/ritk/actions/runs/45214173132)

The [manual comparison](features.md#reading-workstation-parity) records the reference behavior. The [fusion backlog item](../../backlog/RITK-SNAP-NATIVE-FUSION-001.md) starts the next comparison-related milestone.
