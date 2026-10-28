<a id="RITK-SNAP-MULTI-SERIES-001"></a>

## RITK-SNAP-MULTI-SERIES-001 — Load and compare multiple DICOM series — todo
- outcome: Load several study series and display them in independently controlled viewports.
- acceptance: A public multi-series fixture populates the browser; each selection shows its own exact pixels, geometry, and navigation state without replacing or cross-wiring other open series.
- scope: crates/ritk-snap/src/{app,session,presentation,ui}/, DICOM fixtures, tests, and manual captures
- next: Checked-in public multi-series fixture on disk (today's studies are synthesized per-run), shared navigation state extracted out of SnapApp into the per-panel model the native session already uses, and manual captures.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: todo
- needs: none
- priority: feature
