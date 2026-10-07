<a id="RITK-MGH-CONVERSION-001"></a>

## RITK-MGH-CONVERSION-001 — Preserve MGH and MGZ volume semantics — blocked
- outcome: convert MGH/MGZ volumes through RITK without changing represented samples or geometry.
- acceptance: supported scalar types, affine geometry, calibration, and compressed/uncompressed packaging round-trip or return typed loss before output changes.
- scope: crates/ritk-mgh/, crates/ritk-io/, MGH tests and manual
- next: After inventory, implement the MGH/MGZ codec adapter and verify samples, RAS geometry, and compression.
- basis: 4ebc650d6e25a7a6775910ba23bc35c8c7cb78e4
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-MGH-CONVERSION-INVENTORY-001
- priority: correctness
