<a id="RITK-SNAP-MULTI-SERIES-001"></a>

## RITK-SNAP-MULTI-SERIES-001 — Load and compare multiple DICOM series — blocked
- outcome: Load several study series and display them in independently controlled viewports.
- acceptance: A public multi-series fixture populates the browser; each selection shows its own exact pixels, geometry, and navigation state without replacing or cross-wiring other open series.
- scope: crates/ritk-snap/src/{app,session,presentation,ui}/, DICOM fixtures, tests, and manual captures
- next: Connect the RITK study catalog to separate stored series and verify selection and pane isolation.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-DICOM-STORED-IMPORT-001
- priority: feature
