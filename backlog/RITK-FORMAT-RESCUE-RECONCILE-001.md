<a id="RITK-FORMAT-RESCUE-RECONCILE-001"></a>

## RITK-FORMAT-RESCUE-RECONCILE-001 — Rehome format work from stale rescue branches — blocked
- outcome: Re-derive preserved format changes under their current RITK owners.
- acceptance: Every stale format rescue's unique work is delivered through its owner item or remains explicitly preserved there; close rescue PRs only after exact landed-work proof.
- scope: format rescue PRs and their owning RITK codec, adapter, and test paths
- next: After format owners land, re-derive each rescue residual and close only emptied refs by ancestry proof.
- basis: 4ebc650d6e25a7a6775910ba23bc35c8c7cb78e4
- status: blocked
- blocker: Preserved format changes must land through their owning items first.
- needs: RITK-FORMAT-CONVERSION-001
- priority: correctness
