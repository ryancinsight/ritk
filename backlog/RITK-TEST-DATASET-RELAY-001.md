<a id="RITK-TEST-DATASET-RELAY-001"></a>
## RITK-TEST-DATASET-RELAY-001 — Externalize large public test datasets — blocked
- priority: verification; milestone: M-viewer-reading
- outcome: Keep small analytical goldens in Git and fetch larger public datasets through the existing checksummed external-data path.
- acceptance: Every fetch verifies a trusted SHA-256; all test consumers resolve the migrated inputs; a committed fixture-byte budget is enforced; unknown or identifying datasets remain untouched.
- scope: test_data/, externals/, xtask/src/datasets/, consuming tests, and dataset documentation
- next: Expand the shared catalog with verified upstream sources and migrate the remaining consumers. The initial OpenNeuro registration consumer accepts the catalog cache. Retention and removal require a separate decision.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- evidence: The OpenNeuro ds000102 source object and tracked volume agree in all 10,581,116 bytes. A cached fetch passes the inter-subject registration test. The private [Atlas fixtures catalog](https://github.com/ryancinsight/atlas-fixtures/tree/35656a6a21b0b2a5f419fa4fe8fbc4e6fc488a41) contains source records, sizes and hashes. A fresh clone resolves four verified OpenNeuro MRIs. RITK, SimpleITK and NiBabel agree on all 42,991,616 calibrated voxels.
- blocker: Source and redistribution records for the skull CT and paired MRI/CT collections remain unresolved.
- needs: RITK-GAP-2026-08-20-04
