<a id="RITK-FIXTURE-RETENTION-001"></a>

## RITK-FIXTURE-RETENTION-001 — Decide imaging fixture retention — blocked
- outcome: Preserve identifiable or unique imaging data until its owner records a retention decision.
- acceptance: Each inventory row with unknown provenance, license, or identifying metadata has an explicit owner decision before removal or redistribution.
- scope: test_data/, payload manifest, and fixture documentation
- next: After the provenance inventory, ask the owner about each unresolved row; keep all bytes until answered.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: blocked
- blocker: The exact affected files and consumer uses must be inventoried before requesting a retention decision.
- needs: RITK-GAP-2026-08-20-04
- priority: verification
