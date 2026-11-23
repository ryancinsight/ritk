<a id="RITK-METAIMAGE-CONVERSION-001"></a>
## RITK-METAIMAGE-CONVERSION-001 — Preserve MetaImage volume semantics — todo
- priority: correctness; milestone: M-viewer-reading
- outcome: convert MetaImage volumes through RITK without silent sample or metadata loss.
- acceptance: supported MHA/MHD element types, byte order, geometry, and calibration round-trip; unsupported semantics fail preflight before output changes.
- scope: crates/ritk-metaimage/, crates/ritk-io/, MetaImage tests and manual
- next: Implement stored sample decoding and the shared conversion preflight against the [field inventory](../docs/book/metaimage_format.md#conversion-inventory).
- basis: 2c277e2d5b1e2c3b60b2c8afb587ab72d68f690a
- needs: RITK-METAIMAGE-CONVERSION-INVENTORY-001
