<a id="RITK-MINC-CONVERSION-001"></a>

## RITK-MINC-CONVERSION-001 — Preserve MINC volume semantics — blocked
- outcome: Route MINC volumes through stored models and shared image-format dispatch.
- acceptance: Real MINC reads and writes preserve supported samples, geometry, units, calibration, and axes or return typed loss before output changes; hostile dimensions stay bounded.
- scope: crates/ritk-minc/, crates/ritk-io/src/format/minc/, conversion tests, and guide
- next: Add the MINC codec adapter to shared conversion preflight and verify real stored-series round-trips.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-IO-FORMAT-CAPABILITIES-001
- priority: correctness
