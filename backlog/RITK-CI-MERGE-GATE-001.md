<a id="RITK-CI-MERGE-GATE-001"></a>
## RITK-CI-MERGE-GATE-001 — Gate pull requests through one CI check — todo
- milestone: M-viewer-access
- outcome: preserve an accurate required gate across draft and ready pull requests.
- acceptance: one aggregate check covers verification jobs; draft pull requests run no heavy jobs and fail the aggregate; `ready_for_review` runs affected checks; the main ruleset requires only the aggregate; workflow contracts and hosted runs verify Rust, Python, docs, and board-only changes; API checks include the unpublished ritk-snap library and select affected packages/configurations instead of all 31 unrelated listed libraries.
- scope: `.github/workflows/`, `scripts/tests/`, `docs/adr/`, and the main merge ruleset
- next: claim the item, map the required-check and reusable-workflow graph, then record the single-pipeline design in an indexed ADR; close the ci.yml static-package omission for ritk-snap and verify changed-package selection on GUI-only changes.
- basis: c4c271b5b253b0ec92880747dbc67933b96ef6ca
- needs: none
- priority: architecture
