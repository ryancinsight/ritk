<a id="RITK-METAIMAGE-CONVERSION-001"></a>

## RITK-METAIMAGE-CONVERSION-001 — Preserve MetaImage volume semantics — blocked
- outcome: convert MetaImage volumes through RITK without silent sample or metadata loss.
- acceptance: supported MHA/MHD element types, byte order, geometry, and calibration round-trip; unsupported semantics fail preflight before output changes.
- scope: crates/ritk-metaimage/, crates/ritk-io/, MetaImage tests and manual
- next: After the inventory, implement the MetaImage adapter through shared conversion preflight.
- basis: 2c277e2d5b1e2c3b60b2c8afb587ab72d68f690a
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-IO-FORMAT-CAPABILITIES-001, RITK-METAIMAGE-CONVERSION-INVENTORY-001
- priority: correctness
