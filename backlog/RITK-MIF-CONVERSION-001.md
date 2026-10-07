<a id="RITK-MIF-CONVERSION-001"></a>

## RITK-MIF-CONVERSION-001 — Preserve MRtrix image semantics — blocked
- outcome: Connect MIF read/write operations to stored models and shared dispatch.
- acceptance: Supported samples, geometry, units, calibration, diffusion metadata, and axes survive or yield scoped typed loss before output changes.
- scope: crates/ritk-mif/, crates/ritk-io/src/format/mif/, conversion tests, and guide
- next: Add the MIF codec adapter to shared conversion preflight and verify real paths against stored-series semantics.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-IO-FORMAT-CAPABILITIES-001
- priority: correctness
