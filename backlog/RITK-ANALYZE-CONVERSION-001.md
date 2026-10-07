<a id="RITK-ANALYZE-CONVERSION-001"></a>

## RITK-ANALYZE-CONVERSION-001 — Preserve Analyze volume semantics — blocked
- outcome: convert Analyze image/header pairs through RITK without silent sample or geometry changes.
- acceptance: supported scalar types, paired-file identity, byte order, and spatial semantics round-trip or return typed loss before output changes.
- scope: crates/ritk-analyze/, crates/ritk-io/, Analyze tests and manual
- next: After inventory, implement Analyze pair conversion and prove rejection preserves both destinations.
- basis: 2c277e2d5b1e2c3b60b2c8afb587ab72d68f690a
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-ANALYZE-CONVERSION-INVENTORY-001
- priority: correctness
