<a id="RITK-ANALYZE-CONVERSION-001"></a>

## RITK-ANALYZE-CONVERSION-001 — Preserve Analyze volume semantics — todo
- outcome: convert Analyze image/header pairs through RITK without silent sample or geometry changes.
- acceptance: supported scalar types, paired-file identity, byte order, and spatial semantics round-trip or return typed loss before output changes.
- scope: crates/ritk-analyze/, crates/ritk-io/, Analyze tests and manual
- next: Implement the Analyze `StoredSeries` read/write and its `ConversionTarget`/`ConversionAdapter` on the inventory matrix; prove rejection preserves both destinations.
- basis: e0316a43a2f73ddfe5b525d82b7360071fcfa519
- status: todo
- needs: none
- priority: correctness
