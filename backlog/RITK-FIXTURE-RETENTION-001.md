<a id="RITK-FIXTURE-RETENTION-001"></a>
## RITK-FIXTURE-RETENTION-001 — Decide imaging fixture retention — blocked
- priority: verification; milestone: M-viewer-reading
- outcome: Preserve identifiable or unique imaging data until its owner records a retention decision.
- acceptance: Each inventory row with unknown provenance, license, or identifying metadata has an explicit owner decision before removal or redistribution.
- scope: test_data/, payload manifest, and fixture documentation
- decision: [patch] The owner authorizes removing the skull CT and paired MRI/CT collections from new checkouts and retaining both locally. Other unresolved collections still require an individual decision.
- evidence: All 308 skull CT files and 2,334 paired MRI/CT files match their tracked objects at `71f94fc74`. The owner's local inventory records each retained file's size and SHA-256. The private shared catalog contains only verified public fixture records.
- next: Reconcile source records for the remaining tracked collections and request a retention decision where terms remain unresolved.
- basis: 71f94fc7439a15773d48e3cffd156cad1158d4e9
- needs: RITK-GAP-2026-08-20-04
