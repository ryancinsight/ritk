# RITK Gap Audit

<a id="RITK-REGISTRY-PUBLICATION-001"></a>
## RITK-REGISTRY-PUBLICATION-001: Git validation does not establish registry publication [patch]
- risk: a corrected Git revision passes native checks while the registry resolves older incompatible dependencies or lacks required packages.
- evidence: main's release package verification fails in registry Leto 0.42.0; an isolated Coeus 0.11 probe compiles but its registry plan holds registration on unpublished Consus and missing Coeus component versions.
- next: preserve active provider owners, validate the complete required registry graph, and complete approved first publications before claiming a released registration version.
- review: [RITK-REGISTRY-PUBLICATION-001](backlog/RITK-REGISTRY-PUBLICATION-001.md) records exact failure, ownership and credential boundaries.

<a id="PERF-432-01"></a>
## PERF-432-01: Revalidate the legacy MSE performance claim
- risk: June profiling may no longer describe the current backend and can select the wrong optimization target.
- evidence: prior profile claims predate backend and interpolation changes; no current-main reproduction is recorded.
- reopen: before changing MSE or B-spline performance, collect a new exact-main profile and replace or remove the stale claim.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
