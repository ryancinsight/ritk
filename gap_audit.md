# RITK Gap Audit

<a id="RITK-PYTHON-SCRIPT-RUNTIME-001"></a>
## RITK-PYTHON-SCRIPT-RUNTIME-001: Script tests require newer Python APIs [patch]
- risk: supported Python 3.9/3.10 cells stop before wheel/parity/Rust validation; shell PATH changes can select a different interpreter and hide version defects.
- evidence: unchanged main `8af30a9a`; real Python 3.9 reproduces nine failed lock-parser subcases, followed by three gallery `write_text(newline=...)` errors after parser support is restored.
- correction: declared conditional TOML dependency and setup, portable LF writers, and interpreter/format/byte regressions; 41 script tests pass on four real interpreter versions.
- review: [RITK-PYTHON-SCRIPT-RUNTIME-001](backlog/RITK-PYTHON-SCRIPT-RUNTIME-001.md) records the separate draft, final-revision validation criteria and the distinction between draft-policy and executed CI outcomes.

<a id="PERF-432-01"></a>
## PERF-432-01: Revalidate the legacy MSE performance claim
- risk: June profiling may no longer describe the current backend and can select the wrong optimization target.
- evidence: prior profile claims predate backend and interpolation changes; no current-main reproduction is recorded.
- reopen: before changing MSE or B-spline performance, collect a new exact-main profile and replace or remove the stale claim.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
