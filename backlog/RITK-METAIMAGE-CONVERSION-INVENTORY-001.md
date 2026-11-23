<a id="RITK-METAIMAGE-CONVERSION-INVENTORY-001"></a>
## RITK-METAIMAGE-CONVERSION-INVENTORY-001 — Map MetaImage volume semantics — done [patch]
- priority: correctness; milestone: M-viewer-reading
- outcome: Map MetaImage headers and stored samples to RITK volume fields or typed loss.
- acceptance: A checked matrix covers dimension, type, byte order, compression, origin, spacing, direction, detached payload, and calibration with value-test references.
- scope: crates/ritk-metaimage/, crates/ritk-io/src/format/metaimage/, tests, and manual
- evidence: The [conversion inventory](../docs/book/metaimage_format.md#conversion-inventory) maps sample types, packaging, geometry, and unsupported metadata. All 25 MetaImage library tests pass.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- needs: none
