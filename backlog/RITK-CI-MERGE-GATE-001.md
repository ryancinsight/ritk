<a id="RITK-CI-MERGE-GATE-001"></a>
## RITK-CI-MERGE-GATE-001 — Gate pull requests through one CI check — todo
- priority: architecture; milestone: M-viewer-access
- outcome: preserve one accurate, bounded required gate across draft and ready pull requests.
- acceptance: one aggregate check covers every selected verification job; drafts run no heavy jobs and fail the aggregate; `ready_for_review` runs affected checks; the ruleset requires only the aggregate; changed paths select affected packages, supported configurations, and reverse consumers for Rust tests and API checks, including unpublished `ritk-snap`, without scanning 31 unrelated libraries; Python, documentation, and board-only paths retain their checks; a warm hosted ready-PR run meets the five-minute job target.
- finding: At basis 3e48c8663fbacdf048918f541ff885aaa903dc9b, `.github/workflows/ci.yml:358–364` still exits successfully when expected `ritk-snap` metadata is absent; remove this obsolete return-to-workspace skip and assert missing viewer coverage fails.
- scope: `.github/workflows/ci.yml`, affected-package selection and workflow-contract tests, an indexed CI ADR, and the main merge ruleset
- next: reproduce run 44470367130 from pull request 992 (SemVer API action 478 s; test job 498 s), map its required-check graph, then replace the all-library API scan and repeated package builds with affected-package selection while retaining each selected test, API, Python, documentation, and board contract.
- basis: b2fe302b3659ec1dc69bba01849fab7f0e453d1c
- needs: none
