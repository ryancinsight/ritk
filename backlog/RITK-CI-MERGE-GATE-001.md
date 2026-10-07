<a id="RITK-CI-MERGE-GATE-001"></a>

## RITK-CI-MERGE-GATE-001 — Gate pull requests through one CI check — todo
- outcome: preserve an accurate required gate across draft and ready pull requests.
- acceptance: one aggregate check covers verification jobs; draft pull requests run no heavy jobs and fail the aggregate; `ready_for_review` runs affected checks; the main ruleset requires only the aggregate; workflow contracts and hosted runs verify Rust, Python, docs, and board-only changes.
- scope: `.github/workflows/`, `scripts/tests/`, `docs/adr/`, and the main merge ruleset
- next: claim the item, map the required-check and reusable-workflow graph, then record the single-pipeline design in an indexed ADR before changing workflows.
- basis: bf589f94b9826be8b4b05c85e5da31c338f01bab
- status: todo
- needs: none
- priority: architecture
