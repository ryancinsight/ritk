<a id="RITK-SNAP-ELLIPSE-PIXEL-MASK-001"></a>
## RITK-SNAP-ELLIPSE-PIXEL-MASK-001 — Show ellipse contributing-pixel outlines — todo
- outcome: Holding Ctrl while drawing a magnified ellipse shows the boundary of the exact pixels used for its statistics.
- acceptance: A radius-two ellipse over pixel centres selects the independently enumerated 13-pixel set; transformed native edge commands match that set, disappear on Ctrl release, and preserve annotation values.
- scope: crates/ritk-snap/src/tools/interaction/, crates/ritk-snap/src/presentation/native_session/layout/, native ROI tests
- next: Expose the existing RITK inclusion predicate through one mask operation; use Métis primitives for drawing the resulting boundary. RadiAnt 2021.2 p.82.
- links: [comparison](../docs/manual/features.md#reading-and-annotations); [milestone](../backlog.md#M-viewer-reading)
- needs: none
- priority: feature; milestone: M-viewer-reading
- basis: f6e3164af3ccb839e8e99bb2caf5fc3cb39b3f8d
