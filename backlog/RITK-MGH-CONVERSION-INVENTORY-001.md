<a id="RITK-MGH-CONVERSION-INVENTORY-001"></a>
## RITK-MGH-CONVERSION-INVENTORY-001 — Map MGH volume semantics — done [patch]
- priority: correctness; milestone: M-viewer-reading
- outcome: Map MGH/MGZ headers, geometry, samples, and packaging to RITK volume fields or typed loss.
- acceptance: A checked matrix covers RAS geometry, affine, scalar types, gzip, and .mgh/.mgz routes with value-test references.
- scope: crates/ritk-mgh/, crates/ritk-io/src/format/mgh/, tests, and manual
- evidence: [Field mapping and executable references](../docs/book/mgh_format.md#conversion-inventory). The viewer test uses an independent NiBabel file, and the writer test inspects the serialized RAS center.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- needs: none
