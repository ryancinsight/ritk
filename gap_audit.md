# RITK Gap Audit

## RITK-DEPENDENCY-827 [patch][major] — Coeus API and upstream merge boundary

- Risk: a resolved dependency graph does not establish compatible callers or
  a merge-ready PR. Unqualified fresh updates still depend on Metis #499 landing.
- Evidence: canonical all-feature metadata resolves 880 packages; lock integrity,
  dependency alignment, formatting and 37 Python script tests pass. Coeus 0.12
  returns typed Results where RITK image/model callers assume values. Keep-going
  checking yields 321 distinct diagnostics in 32 files; Clippy, nextest build
  and rustdoc fail. Later dependent crates and Python wheels remain unverified.
- Scope: seven original dependency commits and manifests remain preserved;
  [backlog contract](backlog/RITK-DEPENDENCY-827.md) records the revisions and examples.
- Closure: land the upstream correction, specify public error propagation in an
  ADR, migrate all callers and pass the existing complete CI gates.
- Separate provider risk: Consus's standalone Moirai ^0.7 requirement fails;
  that optional dependency is absent from this RITK graph and belongs to #170.

<a id="PERF-432-01"></a>
## PERF-432-01: Revalidate the legacy MSE performance claim
- risk: June profiling may no longer describe the current backend and can select the wrong optimization target.
- evidence: prior profile claims predate backend and interpolation changes; no current-main reproduction is recorded.
- reopen: before changing MSE or B-spline performance, collect a new exact-main profile and replace or remove the stale claim.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
