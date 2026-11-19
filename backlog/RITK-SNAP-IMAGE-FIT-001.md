<a id="RITK-SNAP-IMAGE-FIT-001"></a>
## RITK-SNAP-IMAGE-FIT-001 — Share continuous sample-aspect fitting — todo
- milestone: M-viewer-reading
- outcome: one reusable Métis image-fit operation serves RITK native layout, Zoom 1:1, and the eframe shell.
- acceptance: square 0.5/1/2 mm samples fit equally; anisotropic quarter-turn extents and Zoom 1:1 remain correct; reject invalid/unrepresentable extents and preserve each consumer arithmetic precision.
- scope: `crates/ritk-snap/src/ui/zoom.rs`, native `layout/geometry.rs`, `app/image_placement.rs`; upstream `metis-ui-lang::image::fit`
- next: extend the existing continuous image-fit family without integer extent rounding; retain DICOM spacing/orientation mapping in RITK and remove the three repeated normalization/fit formulas.
- basis: c4c271b5b253b0ec92880747dbc67933b96ef6ca
- needs: none
- priority: architecture
