<a id="RITK-FIXTURE-RETENTION-001"></a>
## RITK-FIXTURE-RETENTION-001 — Decide imaging fixture retention — blocked
- priority: verification; milestone: M-viewer-reading
- outcome: Preserve identifiable or unique imaging data until its owner records a retention decision.
- acceptance: Each inventory row with unknown provenance, license, or identifying metadata has an explicit owner decision before removal or redistribution.
- scope: test_data/, payload manifest, and fixture documentation
- decision: [patch] The owner authorizes removing the skull CT and paired MRI/CT collections from new checkouts and retaining both locally. Other unresolved collections still require an individual decision.
- evidence: All 308 skull CT files and 2,334 paired MRI/CT files match their tracked objects at `0d0c53537`. The owner's local inventory records each retained file's size and SHA-256. The private shared catalog contains only verified public fixture records.
- next: Reconcile source records for the remaining tracked collections and request a retention decision where terms remain unresolved.
- basis: 0d0c5353731d66b28ba517233ca166d5c89e0bc9
- needs: RITK-GAP-2026-08-20-04
