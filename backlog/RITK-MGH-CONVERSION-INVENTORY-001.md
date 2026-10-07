<a id="RITK-MGH-CONVERSION-INVENTORY-001"></a>

## RITK-MGH-CONVERSION-INVENTORY-001 — Map MGH volume semantics — todo
- outcome: Map MGH/MGZ headers, geometry, samples, and packaging to RITK volume fields or typed loss.
- acceptance: A checked matrix covers RAS geometry, affine, scalar types, gzip, and .mgh/.mgz routes with value-test references.
- scope: crates/ritk-mgh/, crates/ritk-io/src/format/mgh/, tests, and manual
- next: Compare the declared header, reader, series, and writer behavior against the typed conversion contract.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: todo
- needs: none
- priority: correctness
