<a id="RITK-TEST-DATASET-RELAY-001"></a>

## RITK-TEST-DATASET-RELAY-001 — Externalize large public test datasets — blocked
- outcome: Keep small analytical goldens in Git and fetch larger public datasets through the existing checksummed external-data path.
- acceptance: Every fetch verifies a trusted SHA-256; all test consumers resolve the migrated inputs; a committed fixture-byte budget is enforced; unknown or identifying datasets remain untouched.
- scope: test_data/, externals/, xtask/src/datasets/, consuming tests, and dataset documentation
- next: Use the payload inventory to source trusted digests, move eligible public data, migrate consumers, and add budget enforcement.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: blocked
- blocker: Provenance and consumer inventory must identify eligible public payloads before any relocation.
- needs: RITK-GAP-2026-08-20-04
- priority: verification
