<a id="RITK-SNAP-ELLIPSE-CONSTRAINT-001"></a>
## RITK-SNAP-ELLIPSE-CONSTRAINT-001 — Constrain ellipse drawing to a circle — todo
- outcome: Holding Shift while drawing an ellipse constrains its displayed radii equally.
- acceptance: Native pointer tests assert circle geometry under anisotropic and rotated pane transforms, modifier release, unchanged slice ownership, cancellation, and analytically recomputed values.
- scope: crates/ritk-snap/src/app/pointer_ops/, crates/ritk-snap/src/presentation/native_session/, upstream GUI aspect constraints
- next: Inspect existing Métis drag constraints; put reusable aspect mechanics there and image geometry in RITK. RadiAnt 2021.2 p.82.
- links: [comparison](../docs/manual/features.md#reading-and-annotations); [milestone](../backlog.md#M-viewer-reading)
- needs: none
- priority: feature; milestone: M-viewer-reading
- basis: f6e3164af3ccb839e8e99bb2caf5fc3cb39b3f8d
